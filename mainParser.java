package dependency_parser;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.Word;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;


public class mainParser {

	public static void main(String[] args) throws IOException {
		String grammar = args.length > 0 ? args[0] : "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
		String[] options = { "-maxLength", "80", "-retainTmpSubcategories" };
		LexicalizedParser lp = LexicalizedParser.loadModel(grammar, options);
		TreebankLanguagePack tlp = lp.getOp().langpack();
		GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
		System.out.println("Pass1");
		Iterable<List<? extends HasWord>> sentences;

		String File="test_src_preprocesed.txt";
		BufferedReader br = null;
		FileReader fr = null;
		PrintWriter writer = new PrintWriter("dependencies_test.txt", "UTF-8");


		try {

			//br = new BufferedReader(new FileReader(FILENAME));
			fr = new FileReader(File);
			br = new BufferedReader(fr);

			String sCurrentLine;
			int count=0;
			while ((sCurrentLine = br.readLine()) != null) {
				//				System.out.println(sCurrentLine);
				Tokenizer<? extends HasWord> toke =tlp.getTokenizerFactory().getTokenizer(new StringReader(sCurrentLine));
				List<? extends HasWord> sentence = toke.tokenize();

				Tree parse = lp.parse(sentence);
//				System.out.println("Sentence : "+sentence);
				GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
				List<TypedDependency> tdl = gs.typedDependenciesCCprocessed();
				writer.println(tdl);
				count+=1;
				System.out.println(count);
				//	    System.out.println(parse.taggedYield()); \\Tags
			}

		} catch (IOException e) {

			e.printStackTrace();
		}
		writer.close();
	}
	//
	//	  private mainParser() {} // static methods only

}


