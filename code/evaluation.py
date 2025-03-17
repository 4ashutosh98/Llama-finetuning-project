from bert_score import score
from sacrebleu.metrics import BLEU
from rouge import Rouge

class SimilarityEvaluator:
    def bert_score_calc(cands, ref1):
        P1, R1, F1 = score(cands, ref1, lang="en", verbose=True)
        return F1

    def bleu_score_calc(cands, ref1):
        bleu_scorer = BLEU()
        score1 = bleu_scorer.sentence_score(
            hypothesis=cands[0],
            references=ref1,
        )
        return score1.score/100

    def rouge_score_calc(cands, ref1):
        rouge_scorer = Rouge()
        score1 = rouge_scorer.get_scores(
            hyps=cands[0],
            refs=ref1[0],
        )
        # return score1[0]['rouge-1']['f']
        return score1[0]['rouge-l']['f']

    def evaluate_similarity(self, cands, ref1):
        bert_score = self.bert_score_calc(cands, ref1)
        bleu_score = self.bleu_score_calc(cands, ref1)
        rouge_score = self.rouge_score_calc(cands, ref1)
        return (bert_score, bleu_score, rouge_score)
