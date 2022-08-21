from typing import Iterable, Dict, List
import numpy as np
from gensim.models import KeyedVectors
import argparse
import os
import warnings
import json


def make_one_concept(model: KeyedVectors,
                     word_list: Iterable[str],
                     concept_name: str = "",
                     normalize: bool = True) -> np.ndarray:
    """
    Create a concept vector from a list of words.
    :param model: Word embedding model. KeyedVectors object.
    :param word_list: List of words describing the concept.
    :param concept_name: (Optional) Name of the concept.
    :param normalize: Whether to normalize the concept vector by its l2 norm (True) or not (False).
    :return: A d-dimensional concept vector aggregated from the vectors in wordlist.
    """
    dim = model.vector_size
    concept_vector = np.zeros(dim, dtype=float)
    word_list = list(set(word_list))
    count = 0
    for w in word_list:
        if w not in model:
            continue
        concept_vector += model[w]
        count += 1
    if count == 0:
        warnings.warn(f"No word in concept '{concept_name}' found in embedding model.")
    if normalize is True and count > 0:
        concept_vector /= np.linalg.norm(concept_vector)
    return concept_vector


def make_concepts_from_lexicon(model: KeyedVectors,
                               lexicon: Dict[str, List[str]],
                               normalize: bool = True,
                               verbose: bool = True) -> Dict[str, np.ndarray]:
    """
    Create a dictionary of concept vectors form a lexicon and embedding model.
    :param model: Word embedding model. KeyedVectors object.
    :param lexicon: Lexicon in {concept: [word1, word2]} format.
    :param normalize: Whether to normalize the concept vector by its l2 norm (True) or not (False).
    :param verbose: Whether to print messages (True) or not (False).
    :return: A d-dimensional concept vector aggregated from the vectors in wordlist.
    """
    concepts = {}
    for concept, words in lexicon.items():
        if verbose:
            print(f"Creating vector for concept {concept}...")
        concepts[concept] = make_one_concept(model=model,
                                             word_list=words,
                                             normalize=normalize,
                                             concept_name=concept)
    return concepts


def parse_args():
    parser = argparse.ArgumentParser(description='Create concepts from word embeddings and lexicons')
    parser.add_argument("--lexicon_path",
                        type=str,
                        help="Path to the lexicon JSON file. Must be in {concept: [word1, word2]} format.",
                        required=True)
    parser.add_argument("--embedding_path",
                        type=str,
                        help="Path to the word embedding file. Must be in word2vec format.",
                        required=True,
                        )
    parser.add_argument("--output_path",
                        type=str,
                        help="Path to the output file, in JSON {concept1: vector1, concept2: vector2} format.",
                        required=True,
                        )
    parser.add_argument("--normalize",
                        type=int,
                        help="Whether to normalize the concept vectors by their l2 norms (1) or not (0).",
                        default=1)
    parser.add_argument("--verbose",
                        type=int,
                        help="Whether to print messages (1) or not (0).",
                        default=1)
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Check files' existence
    assert os.path.isfile(args.lexicon_path), "Lexicon file not found"
    assert os.path.isfile(args.embedding_path), "Embedding file not found"
    if os.path.isfile(args.output_path):
        warnings.warn("Output file already exists. It will be overwritten.")

    verbose = bool(args.verbose)
    if verbose:
        print("Loading lexicon...")
    lexicon = json.load(open(args.lexicon_path))

    if verbose:
        print("Loading word embedding model...")
    embedding = KeyedVectors.load_word2vec_format(args.embedding_path, binary=False)

    if verbose:
        print("Creating concepts...")
    concepts = make_concepts_from_lexicon(model=embedding,
                                          lexicon=lexicon,
                                          normalize=bool(args.normalize),
                                          verbose=verbose)
    for concept, vector in concepts.items():
        concepts[concept] = list(vector)

    if verbose:
        print("Saving concepts...")
    with open(args.output_path, 'w') as f:
        json.dump(concepts, f, indent="\t")

    if verbose:
        print("Done.")
