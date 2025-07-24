import ast

import nltk
from nltk import word_tokenize
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab")
nltk.download('averaged_perceptron_tagger_eng')

import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, logging

logging.set_verbosity_error()

class MetadataProcessor:
    def __init__(
        self,
        abs_summarizer_version="Falconsai/text_summarization",
        ext_summarizer_version="paraphrase-MiniLM-L6-v2",
        summarizer_type="ext",
        device="cpu",
    ):
        if summarizer_type=="abs":
            self.abs_summarizer_version = abs_summarizer_version
            self.abs_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.abs_summarizer_version
            )
            self.abs_tokenizer = AutoTokenizer.from_pretrained(self.abs_summarizer_version)
            self.abs_summarizer = pipeline(
                "summarization",
                model=self.abs_model,
                tokenizer=self.abs_tokenizer,
                device="cuda",
            )
        else:
            self.ext_summarizer_version = ext_summarizer_version
            self.ext_model = SentenceTransformer(self.ext_summarizer_version, device=device)

    def _abstractive_summary(self, text, summarizer):
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary

    def _extractive_summary(self, sentences, summary_size=5, embedder=None, max_iter=100):
        # Compute sentence embeddings
        if embedder is None:
            embedder = SentenceTransformer(self.ext_summarizer_version)

        embeddings = embedder.encode(sentences)

        # Build similarity matrix
        n = len(sentences)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    sim_matrix[i][j] = np.dot(embeddings[i], embeddings[j])

        # Build graph and rank sentences
        nx_graph = nx.from_numpy_array(sim_matrix)
        try:
            scores = nx.pagerank(nx_graph, max_iter=max_iter, alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            print("[Warning] PageRank failed to converge — using fallback.")
            scores = {i: 1.0 for i in range(n)}  # Uniform fallback
        
        ranked_sentences = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True
        )

        selected = [s for _, s in ranked_sentences[:summary_size]]
        return " ".join(selected)

    def _get_consecutive_segment_groups(self, segments, max_duration):
        """
        Given a list of segments (each with 'start' and 'end' keys),
        return a list of groups of one, two, or three consecutive segments whose
        total duration is less than or equal to max_duration seconds.

        Each group is returned as a list of two elements:
            [ (indices_tuple), total_duration ]

        Parameters:
            segments (list of dict): Each dict must have:
                - 'start': float, start time in seconds.
                - 'end': float, end time in seconds.
            max_duration (float): Maximum allowed total duration for a group.

        Returns:
            list: A list of groups (each group is a two-element list as described above).
        """
        groups = []
        n = len(segments)

        # Loop through each segment as a potential starting point.
        for i in range(n):
            # Group of 1 segment: just segment i.
            dur1 = segments[i]["end"] - segments[i]["start"]
            if dur1 <= max_duration:
                groups.append([(i,), dur1])

            # Group of 2 consecutive segments: segments i and i+1.
            if i + 1 < n:
                dur2 = (segments[i]["end"] - segments[i]["start"]) + (
                    segments[i + 1]["end"] - segments[i + 1]["start"]
                )
                if dur2 <= max_duration:
                    groups.append([(i, i + 1), dur2])

            # Group of 3 consecutive segments: segments i, i+1, and i+2.
            if i + 2 < n:
                dur3 = (
                    (segments[i]["end"] - segments[i]["start"])
                    + (segments[i + 1]["end"] - segments[i + 1]["start"])
                    + (segments[i + 2]["end"] - segments[i + 2]["start"])
                )
                if dur3 <= max_duration:
                    groups.append([(i, i + 1, i + 2), dur3])

            if i + 3 < n:
                dur4 = (
                    (segments[i]["end"] - segments[i]["start"])
                    + (segments[i + 1]["end"] - segments[i + 1]["start"])
                    + (segments[i + 2]["end"] - segments[i + 2]["start"])
                    + (segments[i + 3]["end"] - segments[i + 3]["start"])
                )
                if dur4 <= max_duration:
                    groups.append([(i, i + 1, i + 2, i + 3), dur4])

            if i + 4 < n:
                dur5 = (
                    (segments[i]["end"] - segments[i]["start"])
                    + (segments[i + 1]["end"] - segments[i + 1]["start"])
                    + (segments[i + 2]["end"] - segments[i + 2]["start"])
                    + (segments[i + 3]["end"] - segments[i + 3]["start"])
                    + (segments[i + 4]["end"] - segments[i + 4]["start"])
                )
                if dur5 <= max_duration:
                    groups.append([(i, i + 1, i + 2, i + 3, i + 4), dur5])

            if i + 5 < n:
                dur6 = (
                    (segments[i]["end"] - segments[i]["start"])
                    + (segments[i + 1]["end"] - segments[i + 1]["start"])
                    + (segments[i + 2]["end"] - segments[i + 2]["start"])
                    + (segments[i + 3]["end"] - segments[i + 3]["start"])
                    + (segments[i + 4]["end"] - segments[i + 4]["start"])
                    + (segments[i + 5]["end"] - segments[i + 5]["start"])
                )
                if dur6 <= max_duration:
                    groups.append([(i, i + 1, i + 2, i + 3, i + 4, i + 5), dur6])

        return groups

    def get_text_prompt(self, descriptions, stem, fast=True, minlen=24, maxlen=512):
        # created a list of neg words, removing the sentences that any of these include these
        if stem == "vocal":
            neg_words = [
                "low",
                "mono",
                "noisy",
                "amateur",
                "bad",
                "poor",
                "no",
                "not",
                "quality",
            ]  # can add "noises", "foreign"
        else:
            neg_words = [
                "low",
                "male",
                "female",
                "mono",
                "noisy",
                "amateur",
                "vocals",
                "voices",
                "bad",
                "poor",
                "no",
                "not",
                "quality",
            ]
        descriptions = ast.literal_eval(descriptions)
        descriptions = " ".join([v["text"] for k, v in descriptions.items()])
        descriptions = sent_tokenize(descriptions)
        descriptions = list(
            set(
                [
                    desc
                    for desc in descriptions
                    if len(set(word_tokenize(desc)).intersection(set(neg_words))) == 0
                ]
            )
        )

        # return empty string
        if len(descriptions) == 0:
            return ""

        if fast:
            summarized = self._extractive_summary(descriptions, 5, self.ext_model)
        else:
            ext_summary = self._extractive_summary(descriptions, 5, self.ext_model)
            summarized = self._abstractive_summary(
                ext_summary, max_length=maxlen, min_length=minlen, do_sample=False
            )
        return summarized

mproc = MetadataProcessor()

def get_custom_metadata(info, audio):
    prompt = mproc.get_text_prompt(info["prompt"], "accompaniment")
    # Use relative path as the prompt
    return {"prompt": prompt}