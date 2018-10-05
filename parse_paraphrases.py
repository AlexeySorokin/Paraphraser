import sys
import os

from ufal_udpipe import Model
sys.path.append("data/DeepPavlov/deeppavlov")
from deeppavlov.deep import find_config
from deeppavlov.core.commands.infer import build_model_from_config

from read import read_data
from work import MixedUDPreprocessor

PARAPHRASES_DIR = "paraphraser"
INFILES = ["paraphrases_gold.xml"]
OUTFILES = ["parsed_paraphrases_gold_new.xml"]

if __name__ == "__main__":
    tagger = build_model_from_config(find_config("morpho_ru_syntagrus_train_pymorphy"))
    print("Tagger built")
    ud_model = Model.load("russian-syntagrus-ud-2.0-170801.udpipe")
    print("UD Model loaded")
    ud_processor = MixedUDPreprocessor(ud_model, tagger)

    infiles = [os.path.join(PARAPHRASES_DIR, x) for x in INFILES]
    outfiles = [os.path.join(PARAPHRASES_DIR, x) for x in OUTFILES]
    for infile, outfile in zip(infiles, outfiles):
        read_data(infile, from_parses=False, save_file=outfile, ud_processor=ud_processor)