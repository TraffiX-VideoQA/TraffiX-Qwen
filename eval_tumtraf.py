import re
import argparse
import json
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
import ast
import sys
import random
from tabulate import tabulate  # For pretty-printing tables

sys.path.append(".")

# -----------------------------------------------------------------------------
# This script evaluates TUMTraf-QA predictions.
# It uses per-type auto-scoring logic and fills the global
# accuracy/match pools so the overall numbers are correct.
# -----------------------------------------------------------------------------


class TUMTraf_evaluation:
    """Evaluation core – automatically decides accuracy/match per question type."""

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def load_json(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def __init__(self, pred_path: str, gt_path: str, qa_type_path: str):
        self.prediction = self.load_json(pred_path)
        self.GT_qa = self.load_json(gt_path)
        self.qa_type = self.load_json(qa_type_path)

        # Filter predictions so that every id exists in GT
        gt_ids = {item["id"] for item in self.GT_qa}
        self.prediction = [p for p in self.prediction if p["id"] in gt_ids]

        # Initialize containers for all question types
        self.all_question_types = set()
        for _, v in self.qa_type.items():
            if isinstance(v, list):
                self.all_question_types.update(v)
        self.type_data = {t: {"id": [], "answer": [], "GT": []} for t in self.all_question_types}

        # Global pools for each metric type
        self.accuracy = {"id": [], "answer": [], "GT": []}
        self.match = {"id": [], "answer": [], "GT": []}
        self.language = {"id": [], "answer": [], "GT": []}
        self.spatial_temporal = {"id": [], "answer": [], "GT": []}

        self.merge_types = self.qa_type.get("class_merge", {})
        self.type_scoring: dict[str, str] = {}
        self.acc_metrics, self.match_metrics = set(), set()

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def evaluate_language_metrics(self, refs, preds):
        bleu = Bleu(4).compute_score(refs, preds)[0]
        rouge = Rouge().compute_score(refs, preds)[0]
        cider = Cider().compute_score(refs, preds)[0]
        return {
            f"BLEU-{i+1}": bleu[i] for i in range(4)
        } | {"ROUGE_L": rouge, "CIDEr": cider, "METEOR": 0.0, "SPICE": 0.0}

    @staticmethod
    def parse_choice(text: str, choices):
        """Heuristic: find all occurrences and take the last one; fall back to the first plain match; else random."""
        resp = text.strip()
        # Remove common punctuation at both ends
        for ch in [",", ".", "!", "?", ";", ":", "'"]:
            resp = resp.strip(ch)

        cands = [c for c in choices if f"{c}." in resp or resp.strip().upper() == c]
        if not cands:
            return random.choice(choices)
        if len(cands) == 1:
            return cands[0]

        # If multiple, choose the last occurrence in the string
        idxs = [resp.rfind(f" {c} ") for c in cands]
        return cands[int(np.argmax(idxs))]

    def eval_acc(self, data):
        """Evaluate accuracy for multiple-choice questions (A/B/C/D)."""
        choices = ["A", "B", "C", "D"]
        hit = sum(
            self.parse_choice(a, choices) == self.parse_choice(g, choices)
            for a, g in zip(data["answer"], data["GT"])
        )
        return hit / len(data["answer"]) if data["answer"] else 0.0

    def eval_match(self, data):
        """Evaluate match for yes/no questions."""
        hit = 0
        for a, g in zip(data["answer"], data["GT"]):
            a_flag = re.search(r"\b(yes|no)\b", a.lower())
            g_flag = re.search(r"\b(yes|no)\b", g.lower())
            if a_flag and g_flag and a_flag.group(1) == g_flag.group(1):
                hit += 1
        return hit / len(data["answer"]) if data["answer"] else 0.0

    # Parse coordinate tuples for spatial-temporal metrics
    def parse_tuples(self, entry):
        """Parse the coordinate tuple from a given entry string. Returns a tuple of parsed coordinates or None if parsing fails."""
        try:
            coords = ast.literal_eval(re.sub(r'(c\d+)', r'"\1"', entry))
            return tuple(map(float, [coords[0][1], coords[0][2], coords[0][3]])), \
                tuple(map(float, [coords[1][1], coords[1][2], coords[1][3]]))
        except:
            # If parsing fails, try regex extraction
            matches = re.findall(r'[\[\(]?(?:c\d+,)?\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)[\]\)]?', entry)
            if len(matches) == 2:
                return tuple(map(float, [matches[0][1], matches[0][2], matches[0][3]])), \
                    tuple(map(float, [matches[1][1], matches[1][2], matches[1][3]]))
            else:
                return None

    def eval_spatiotemporal(self, data):
        """Evaluate spatial-temporal errors for prediction and ground truth tuples."""
        gt_list = data["GT"]
        pred_list = data["answer"]
        temporal_errors = []
        spatial_errors = []
        spatial_temporal_errors = []

        for pred, gt in zip(pred_list, gt_list):
            gt_coords = self.parse_tuples(gt)
            pred_coords = self.parse_tuples(pred)

            # If parsing fails, assign max error
            if gt_coords and pred_coords:
                (gt_time_start, gt_x_start, gt_y_start), (gt_time_end, gt_x_end, gt_y_end) = gt_coords
                (pred_time_start, pred_x_start, pred_y_start), (pred_time_end, pred_x_end, pred_y_end) = pred_coords
            else:
                temporal_errors.append(1.0)
                spatial_errors.append(1.0)
                spatial_temporal_errors.append(1.0)
                continue

            start_time_error = abs(pred_time_start - gt_time_start)
            end_time_error = abs(pred_time_end - gt_time_end)

            start_position_error = np.sqrt((pred_x_start - gt_x_start)**2 + (pred_y_start - gt_y_start)**2)
            end_position_error = np.sqrt((pred_x_end - gt_x_end)**2 + (pred_y_end - gt_y_end)**2)

            start_error = np.sqrt((pred_x_start - gt_x_start)**2 + (pred_y_start - gt_y_start)**2 + (pred_time_start - gt_time_start)**2)
            end_error = np.sqrt((pred_x_end - gt_x_end)**2 + (pred_y_end - gt_y_end)**2 + (pred_time_end - gt_time_end)**2)

            average_temporal_L1_error = (start_time_error + end_time_error) / 2
            average_space_L2_error = (start_position_error + end_position_error) / 2
            average_space_temporal_L2_error = (start_error + end_error) / 2

            temporal_errors.append(min(1.0, average_temporal_L1_error))
            spatial_errors.append(min(1.0, average_space_L2_error))
            spatial_temporal_errors.append(min(1.0, average_space_temporal_L2_error))

        temporal_error = np.mean(temporal_errors) if temporal_errors else 0.0
        spatial_error = np.mean(spatial_errors) if spatial_errors else 0.0
        spatial_temporal_error = np.mean(spatial_temporal_errors) if spatial_temporal_errors else 0.0

        return {"temporal_error": temporal_error, "spatial_error": spatial_error, "spatial_temporal_error": spatial_temporal_error}

    # Decide scoring rule for a whole type (accuracy or match)
    def get_type_scoring(self, gt_list):
        yes_no = sum(bool(re.search(r"\b(yes|no)\b", g.lower())) for g in gt_list)
        abcd = sum(bool(re.match(r"^[abcd]\b", g.lower())) for g in gt_list)
        return "match" if yes_no > abcd else "accuracy"

    # ------------------------------------------------------------------
    # Data loading – fills global pools so overall accuracy/match work
    # ------------------------------------------------------------------
    def load(self):
        pred_map = {p["id"]: p for p in self.prediction}
        acc_match_cfg = self.qa_type.get("acc_match_metrics", [])

        for qa in self.GT_qa:
            qid, q_type = qa["id"], qa["type"]
            if qid not in pred_map:
                continue
            ans = pred_map[qid]["answer"]
            gt = qa["conversations"][1]["value"]

            bucket = self.type_data.setdefault(q_type, {"id": [], "answer": [], "GT": []})
            for key, v in zip(["id", "answer", "GT"], [qid, ans, gt]):
                bucket[key].append(v)

            if q_type in self.qa_type.get("language_metrics", []):
                for k in ("id", "answer", "GT"):
                    self.language[k].append(bucket[k][-1])
            elif q_type in self.qa_type.get("spatial_temporal_metrics", []):
                for k in ("id", "answer", "GT"):
                    self.spatial_temporal[k].append(bucket[k][-1])
            elif q_type in acc_match_cfg:
                rule = self.get_type_scoring(bucket["GT"])
                self.type_scoring[q_type] = rule
                if rule == "accuracy":
                    score = self.eval_acc(bucket)
                    self.acc_metrics.add(q_type)
                    # Add to global accuracy pool
                    for k in ("id", "answer", "GT"):
                        self.accuracy[k].extend(bucket[k])
                else:
                    score = self.eval_match(bucket)
                    self.match_metrics.add(q_type)
                    # Add to global match pool
                    for k in ("id", "answer", "GT"):
                        self.match[k].extend(bucket[k])
                bucket.update({
                    "score": score,
                    "correct": int(score * len(bucket["id"])),
                    "total": len(bucket["id"]),
                })
            else:
                print("Unknown type", q_type)

    # ------------------------------------------------------------------
    # Merge question types for main class metrics
    # ------------------------------------------------------------------
    def merge_question_types(self):
        for big, childs in self.merge_types.items():
            merged = {"id": [], "answer": [], "GT": []}
            for c in childs:
                for k in merged:
                    merged[k].extend(self.type_data.get(c, {}).get(k, []))
            self.type_data[big] = merged
            rule = self.get_type_scoring(merged["GT"])
            self.type_scoring[big] = rule
            if rule == "accuracy":
                score = self.eval_acc(merged)
            else:
                score = self.eval_match(merged)
            merged.update({
                "score": score,
                "correct": int(score * len(merged["id"])),
                "total": len(merged["id"]),
            })
            # Do not add to global pool to avoid double counting (children already included)

    # ------------------------------------------------------------------
    # Main evaluation logic
    # ------------------------------------------------------------------
    def evaluation(self):
        self.merge_question_types()

        result = {
            "accuracy": self.eval_acc(self.accuracy) if self.accuracy["id"] else 0.0,
            "match":    self.eval_match(self.match) if self.match["id"] else 0.0,
            "language": self.evaluate_language_metrics(
                {str(i): [g] for i, g in enumerate(self.language["GT"])},
                {str(i): [a] for i, a in enumerate(self.language["answer"])}
            ) if self.language["id"] else {},
            "spatial_temporal": self.eval_spatiotemporal(self.spatial_temporal) if self.spatial_temporal["id"] else {},
            "per_type": {},
        }

        for t, bucket in self.type_data.items():
            metric = {"num_questions": len(bucket["id"])}
            if t in self.qa_type.get("language_metrics", []):
                metric |= self.evaluate_language_metrics(
                    {str(i): [g] for i, g in enumerate(bucket["GT"])},
                    {str(i): [a] for i, a in enumerate(bucket["answer"])}
                )
            elif t in self.type_scoring:
                metric |= {
                    "score": bucket["score"],
                    "correct": bucket["correct"],
                    "total": bucket["total"],
                }
            elif t in self.qa_type.get("spatial_temporal_metrics", []):
                metric |= self.eval_spatiotemporal(bucket)
            result["per_type"][t] = metric
        return result


# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-path", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--qa-type-path", required=True)
    args = parser.parse_args()

    ev = TUMTraf_evaluation(args.prediction_path, args.gt_path, args.qa_type_path)
    ev.load()
    out = ev.evaluation()

    outfile = args.prediction_path.replace(".json", "_evaluation_output.txt")
    # 1. Language metrics table
    language_types = ev.qa_type.get("language_metrics", [])
    language_table = []
    for t in language_types:
        m = out["per_type"].get(t, {})
        row = [t]
        for metric in ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE_L", "CIDEr", "METEOR", "SPICE"]:
            row.append(f"{m.get(metric, 0):.4f}" if metric in m else "-")
        language_table.append(row)
    language_headers = ["Type", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE_L", "CIDEr", "METEOR", "SPICE"]

    # 2. Spatial-temporal metrics table
    spatial_types = ev.qa_type.get("spatial_temporal_metrics", [])
    spatial_table = []
    for t in spatial_types:
        m = out["per_type"].get(t, {})
        row = [t]
        for metric in ["temporal_error", "spatial_error", "spatial_temporal_error"]:
            row.append(f"{m.get(metric, 0):.4f}" if metric in m else "-")
        spatial_table.append(row)
    spatial_headers = ["Type", "temporal_error", "spatial_error", "spatial_temporal_error"]

    # 3. QA metrics table (excluding language/spatial-temporal, Description, and main classes)
    exclude_types = {"description", "referredobjectdescription", "spatialtemporalgrounding", "spatial-temporal grounding", "Spatial-Temporal Grounding",
                     "motion", "class", "positioning", "existence", "counting"}
    qa_types = [t for t in out["per_type"] if t not in language_types and t not in spatial_types and t.lower() not in exclude_types]
    qa_table = []
    for t in qa_types:
        m = out["per_type"][t]
        summary = (
            f"{m['score']:.4f} ({m['correct']}/{m['total']})" if "score" in m else
            "; ".join(f"{k}:{v:.4f}" for k, v in m.items() if k not in {"num_questions"}) or "-"
        )
        qa_table.append([t, m["num_questions"], summary])

    # 4. Main classes metrics table
    main_classes = ["Motion", "Class", "Positioning", "Existence", "Counting"]
    class_merge = ev.qa_type.get("class_merge", {})
    per_type_scores = {t: out["per_type"][t]["score"] for t in out["per_type"] if "score" in out["per_type"][t]}
    main_table = []
    main_scores = []
    for big in main_classes:
        children = class_merge.get(big, [])
        child_scores = [per_type_scores.get(c) for c in children if c in per_type_scores]
        if child_scores:
            score = sum(child_scores) / len(child_scores)
            main_scores.append(score)
            # Count total questions and correct answers for the main class
            total = sum(out["per_type"][c]["num_questions"] for c in children if c in out["per_type"])
            correct = sum(out["per_type"][c]["correct"] for c in children if c in out["per_type"] and "correct" in out["per_type"][c])
            main_table.append([big, total, f"{score:.4f} ({correct}/{total})"])
        else:
            main_table.append([big, 0, "-"])
    mean_score = sum(main_scores) / len(main_scores) if main_scores else 0.0
    main_table.append(["Mean", "-", f"{mean_score:.4f}"])

    # Output
    with open(outfile, "w", encoding="utf-8") as f:
        def print_and_write(text):
            print(text)
            f.write(text + "\n")

        print_and_write("================ Language Metrics ================" )
        print_and_write(tabulate(language_table, headers=language_headers, tablefmt="grid") + "\n")

        print_and_write("================ Spatial-Temporal Metrics ================" )
        print_and_write(tabulate(spatial_table, headers=spatial_headers, tablefmt="grid") + "\n")

        print_and_write("================ QA Metrics ================" )
        print_and_write(tabulate(qa_table, headers=["Type", "#Q", "Result"], tablefmt="grid") + "\n")

        print_and_write("================ Main QA Metrics ================" )
        print_and_write(tabulate(main_table, headers=["Type", "#Q", "Result"], tablefmt="grid") + "\n")

    print("Saved", outfile)
