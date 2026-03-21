"""Training Room — ห้องฝึกอบรม AI Agent

สร้าง dataset → เทรน (fine-tune) → ทดสอบ → วัดผล → ปรับปรุง
เหมือนห้องฝึกพนักงานใหม่ของบริษัท

ใช้งาน:
    trainer = TrainingRoom()

    # 1. สร้าง dataset จากตัวอย่างงาน
    trainer.create_dataset("game-coder", [
        {"input": "สร้างระบบ inventory", "output": "class Inventory:..."},
        {"input": "เขียน player movement", "output": "def move(self):..."},
    ])

    # 2. สร้าง dataset จาก AI ตัวเก่ง (distillation)
    trainer.distill("game-coder", teacher="qwen3:14b", topics=[...])

    # 3. เทรน agent ด้วย dataset
    trainer.train("game-coder", dataset="game-coder-training")

    # 4. สอบวัดผล
    trainer.exam("game-coder", questions=[...])
"""

import json
import os
import subprocess
import sys
import time
import re
from pathlib import Path
from datetime import datetime

FACTORY_DIR = Path(__file__).parent
DATASETS_DIR = FACTORY_DIR / "datasets"
EXAMS_DIR = FACTORY_DIR / "exams"
RESULTS_DIR = FACTORY_DIR / "training_results"
REGISTRY_FILE = FACTORY_DIR / "registry.json"


class TrainingRoom:
    """ห้องฝึกอบรม — สอน AI ให้เก่งขึ้นเฉพาะทาง"""

    def __init__(self):
        for d in [DATASETS_DIR, EXAMS_DIR, RESULTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    # ════════════════════════════════════════════════════════════════
    #  1. สร้าง Dataset
    # ════════════════════════════════════════════════════════════════

    def create_dataset(self, name, examples, append=False):
        """สร้าง dataset จากตัวอย่าง input/output

        Args:
            name: ชื่อ dataset
            examples: list of {"input": "...", "output": "..."}
            append: True = เพิ่มต่อจากเดิม
        """
        dataset_path = DATASETS_DIR / f"{name}.jsonl"

        mode = "a" if append else "w"
        with open(dataset_path, mode, encoding="utf-8") as f:
            for ex in examples:
                line = json.dumps({
                    "messages": [
                        {"role": "user", "content": ex["input"]},
                        {"role": "assistant", "content": ex["output"]},
                    ]
                }, ensure_ascii=False)
                f.write(line + "\n")

        total = sum(1 for _ in open(dataset_path, encoding="utf-8"))
        print(f"  Dataset '{name}' saved: {total} examples")
        print(f"  Path: {dataset_path}")
        return dataset_path

    def create_dataset_from_files(self, name, folder_path, file_type=".py"):
        """สร้าง dataset จากไฟล์ code จริง

        อ่านไฟล์ code แล้วสร้าง Q&A pairs:
        - "เขียน function X" → code ของ function X
        - "อธิบาย class Y" → code ของ class Y
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"  Folder not found: {folder_path}")
            return None

        examples = []
        files = list(folder.rglob(f"*{file_type}"))

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            rel_path = file_path.relative_to(folder)

            # สร้าง Q&A: "เขียนไฟล์นี้" → code
            if len(content.strip()) > 50:
                examples.append({
                    "input": f"เขียนไฟล์ {rel_path} ที่ทำหน้าที่ตาม code นี้",
                    "output": content[:4000],  # จำกัดความยาว
                })

            # Extract functions/classes
            for match in re.finditer(r'^(def |class )(\w+)', content, re.MULTILINE):
                kind = "function" if match.group(1).strip() == "def" else "class"
                func_name = match.group(2)
                # หา body
                start = match.start()
                lines = content[start:].split("\n")
                body_lines = [lines[0]]
                for line in lines[1:]:
                    if line and not line[0].isspace() and not line.startswith("#"):
                        break
                    body_lines.append(line)
                body = "\n".join(body_lines)

                if len(body.strip()) > 30:
                    examples.append({
                        "input": f"เขียน {kind} {func_name} ใน {rel_path}",
                        "output": body[:2000],
                    })

        if examples:
            return self.create_dataset(name, examples)
        else:
            print(f"  No examples found in {folder_path}")
            return None

    # ════════════════════════════════════════════════════════════════
    #  2. Distillation — ให้ AI เก่งสอน AI เล็ก
    # ════════════════════════════════════════════════════════════════

    def distill(self, dataset_name, teacher, topics, examples_per_topic=5,
                timeout=180):
        """ให้ model ใหญ่ (teacher) สร้าง training data สำหรับ model เล็ก

        Args:
            dataset_name: ชื่อ dataset ที่จะสร้าง
            teacher: model ที่จะเป็นครู (เช่น "qwen3:14b")
            topics: list of หัวข้อที่อยากให้เก่ง
            examples_per_topic: จำนวนตัวอย่างต่อหัวข้อ
        """
        print(f"\n{'='*60}")
        print(f"  DISTILLATION — Teacher: {teacher}")
        print(f"  Topics: {len(topics)} | Examples/topic: {examples_per_topic}")
        print(f"{'='*60}\n")

        all_examples = []

        for i, topic in enumerate(topics, 1):
            print(f"  [{i}/{len(topics)}] Topic: {topic}")

            for j in range(examples_per_topic):
                # ขั้น 1: ให้ teacher สร้างโจทย์
                question_prompt = (
                    f"สร้างโจทย์/คำถามที่เกี่ยวกับ: {topic}\n"
                    f"ตัวอย่างที่ {j+1}/{examples_per_topic}\n"
                    f"ให้โจทย์มีความยากปานกลาง-สูง ใช้ได้จริง\n"
                    f"ตอบแค่โจทย์อย่างเดียว ไม่ต้องมีคำตอบ"
                )
                q_result = self._ask_model(teacher, question_prompt, timeout)
                if not q_result:
                    continue

                question = q_result.strip()

                # ขั้น 2: ให้ teacher ตอบโจทย์ (เป็นคำตอบตัวอย่าง)
                answer_prompt = (
                    f"ตอบคำถาม/โจทย์นี้อย่างละเอียด:\n\n{question}\n\n"
                    f"ตอบแบบมืออาชีพ ให้ code ที่ใช้ได้จริง พร้อมคำอธิบาย"
                )
                a_result = self._ask_model(teacher, answer_prompt, timeout)
                if not a_result:
                    continue

                all_examples.append({
                    "input": question,
                    "output": a_result.strip(),
                })
                print(f"    Example {j+1}: {question[:60]}...")

        if all_examples:
            self.create_dataset(dataset_name, all_examples)
            print(f"\n  Distillation complete: {len(all_examples)} examples")
        else:
            print(f"\n  ERROR: No examples generated")

        return all_examples

    # ════════════════════════════════════════════════════════════════
    #  3. Train — เทรน Agent ด้วย Dataset
    # ════════════════════════════════════════════════════════════════

    def train(self, agent_name, dataset_name, base_model=None):
        """เทรน agent โดยอัพเดท system prompt ด้วยความรู้จาก dataset

        วิธีการ: อ่าน dataset → สร้าง knowledge summary → ยัดเข้า system prompt
        (Fine-tune จริงต้องใช้ unsloth/axolotl — นี่เป็นวิธี prompt-injection training)

        Args:
            agent_name: ชื่อ agent (ไม่ต้องใส่ "ai-")
            dataset_name: ชื่อ dataset ที่จะใช้เทรน
            base_model: override base model
        """
        full_name = f"ai-{agent_name}" if not agent_name.startswith("ai-") else agent_name
        dataset_path = DATASETS_DIR / f"{dataset_name}.jsonl"

        if not dataset_path.exists():
            print(f"  Dataset not found: {dataset_name}")
            return False

        # อ่าน dataset
        examples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not examples:
            print("  Empty dataset")
            return False

        print(f"\n{'='*60}")
        print(f"  TRAINING: {full_name}")
        print(f"  Dataset: {dataset_name} ({len(examples)} examples)")
        print(f"{'='*60}\n")

        # ดึง modelfile เดิม
        result = subprocess.run(
            ["ollama", "show", full_name, "--modelfile"],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            # agent ยังไม่มี สร้างใหม่
            if not base_model:
                base_model = "qwen3:8b"
            current_system = ""
            from_line = f"FROM {base_model}"
        else:
            modelfile = result.stdout
            # Extract FROM line
            from_match = re.search(r'^FROM (.+)$', modelfile, re.MULTILINE)
            from_line = from_match.group(0) if from_match else "FROM qwen3:8b"

            # Extract current SYSTEM prompt
            sys_match = re.search(r'SYSTEM "(.*?)"', modelfile, re.DOTALL)
            current_system = sys_match.group(1) if sys_match else ""

        # สร้าง training knowledge จาก dataset
        knowledge_lines = []
        for ex in examples[:50]:  # จำกัด 50 ตัวอย่าง (ไม่งั้น prompt ยาวเกิน)
            msgs = ex.get("messages", [])
            if len(msgs) >= 2:
                q = msgs[0]["content"][:200]
                a = msgs[1]["content"][:500]
                knowledge_lines.append(f"Q: {q}\nA: {a}")

        training_knowledge = "\n\n---\n\n".join(knowledge_lines)

        # สร้าง Modelfile ใหม่ที่มี training knowledge
        new_system = current_system.rstrip()
        if training_knowledge:
            new_system += f"""

## Training Knowledge ({len(examples)} examples learned)
คุณได้รับการฝึกอบรมเพิ่มเติมจากตัวอย่างต่อไปนี้ ใช้ความรู้นี้ในการตอบ:

{training_knowledge}
"""

        # Escape quotes in system prompt
        new_system = new_system.replace('"', '\\"')

        new_modelfile = f"""{from_line}

PARAMETER temperature 0.4
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

SYSTEM "{new_system}"
"""

        # Save and build
        modelfile_path = FACTORY_DIR / "agents" / f"{full_name}.Modelfile"
        modelfile_path.parent.mkdir(parents=True, exist_ok=True)
        modelfile_path.write_text(new_modelfile, encoding="utf-8")

        print(f"  Building trained model...")
        result = subprocess.run(
            ["ollama", "create", full_name, "-f", str(modelfile_path)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return False

        # บันทึกผลการเทรน
        train_result = {
            "agent": full_name,
            "dataset": dataset_name,
            "examples_count": len(examples),
            "trained_at": datetime.now().isoformat(),
        }
        result_path = RESULTS_DIR / f"train_{full_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path.write_text(json.dumps(train_result, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"  Training complete!")
        print(f"  Learned: {len(examples)} examples")
        print(f"  Test: ollama run {full_name}")
        return True

    # ════════════════════════════════════════════════════════════════
    #  4. Exam — สอบวัดผล Agent
    # ════════════════════════════════════════════════════════════════

    def exam(self, agent_name, questions=None, exam_file=None, judge=None,
             timeout=120):
        """สอบวัดผล agent — ให้คะแนนอัตโนมัติ

        Args:
            agent_name: ชื่อ agent
            questions: list of {"question": "...", "expected": "...", "points": 10}
            exam_file: path to exam JSON file
            judge: model ที่จะให้คะแนน (default: ใช้ keyword matching)
        """
        full_name = f"ai-{agent_name}" if not agent_name.startswith("ai-") else agent_name

        if exam_file:
            exam_path = Path(exam_file)
            if not exam_path.exists():
                # Try in exams directory
                exam_path = EXAMS_DIR / exam_file
            if exam_path.exists():
                exam_data = json.loads(exam_path.read_text(encoding="utf-8"))
                questions = exam_data.get("questions", [])
            else:
                print(f"  Exam file not found: {exam_file}")
                return None

        if not questions:
            print("  No questions provided")
            return None

        print(f"\n{'='*60}")
        print(f"  EXAM: {full_name}")
        print(f"  Questions: {len(questions)}")
        print(f"  Judge: {judge or 'keyword matching'}")
        print(f"{'='*60}\n")

        total_points = 0
        earned_points = 0
        results = []

        for i, q in enumerate(questions, 1):
            question = q["question"]
            expected = q.get("expected", "")
            keywords = q.get("keywords", [])
            points = q.get("points", 10)
            total_points += points

            print(f"  Q{i}: {question[:70]}...")

            # ให้ agent ตอบ
            response = self._ask_model(full_name, question, timeout)
            if not response:
                print(f"    Score: 0/{points} (no response)")
                results.append({"question": question, "response": "", "score": 0, "max": points})
                continue

            # ให้คะแนน
            if judge:
                score = self._judge_with_ai(judge, question, expected, response, points, timeout)
            else:
                score = self._judge_with_keywords(response, expected, keywords, points)

            earned_points += score
            status = "PASS" if score >= points * 0.6 else "FAIL"
            print(f"    Score: {score}/{points} [{status}]")

            results.append({
                "question": question,
                "response": response[:500],
                "expected": expected[:200],
                "score": score,
                "max": points,
            })

        # สรุปผล
        percentage = (earned_points / total_points * 100) if total_points > 0 else 0
        grade = self._calculate_grade(percentage)

        print(f"\n{'='*60}")
        print(f"  EXAM RESULTS: {full_name}")
        print(f"  Score: {earned_points}/{total_points} ({percentage:.1f}%)")
        print(f"  Grade: {grade}")
        print(f"{'='*60}\n")

        # บันทึกผลสอบ
        exam_result = {
            "agent": full_name,
            "score": earned_points,
            "total": total_points,
            "percentage": percentage,
            "grade": grade,
            "results": results,
            "exam_date": datetime.now().isoformat(),
        }
        result_path = RESULTS_DIR / f"exam_{full_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        result_path.write_text(json.dumps(exam_result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Results saved: {result_path}")

        return exam_result

    def create_exam(self, name, questions):
        """สร้างข้อสอบแล้วบันทึกไว้ใช้ซ้ำ

        Args:
            name: ชื่อข้อสอบ
            questions: list of {"question": "...", "expected": "...", "keywords": [...], "points": 10}
        """
        exam_path = EXAMS_DIR / f"{name}.json"
        exam_data = {
            "name": name,
            "created": datetime.now().isoformat(),
            "questions": questions,
        }
        exam_path.write_text(json.dumps(exam_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Exam '{name}' saved: {len(questions)} questions")
        print(f"  Path: {exam_path}")
        return exam_path

    def auto_generate_exam(self, topic, num_questions=10, teacher=None, timeout=120):
        """ให้ AI สร้างข้อสอบอัตโนมัติ

        Args:
            topic: หัวข้อข้อสอบ
            num_questions: จำนวนข้อ
            teacher: model ที่ใช้สร้างข้อสอบ
        """
        if not teacher:
            teacher = "qwen3:8b"

        prompt = f"""สร้างข้อสอบเรื่อง: {topic}
จำนวน: {num_questions} ข้อ

ตอบเป็น JSON array แบบนี้เท่านั้น (ไม่ต้องมี markdown):
[
  {{"question": "คำถาม", "expected": "คำตอบที่ถูก", "keywords": ["keyword1", "keyword2"], "points": 10}},
  ...
]"""

        print(f"  Generating {num_questions} questions about: {topic}...")
        response = self._ask_model(teacher, prompt, timeout)
        if not response:
            print("  Failed to generate exam")
            return None

        # Parse JSON from response
        try:
            # หา JSON array จาก response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions = json.loads(json_match.group())
            else:
                questions = json.loads(response)

            exam_name = topic.replace(" ", "_").lower()[:30]
            return self.create_exam(exam_name, questions)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  Failed to parse exam: {e}")
            # Save raw response for debugging
            raw_path = EXAMS_DIR / f"raw_{datetime.now().strftime('%H%M%S')}.txt"
            raw_path.write_text(response, encoding="utf-8")
            print(f"  Raw response saved: {raw_path}")
            return None

    # ════════════════════════════════════════════════════════════════
    #  5. Compare — เปรียบเทียบก่อน/หลังเทรน
    # ════════════════════════════════════════════════════════════════

    def compare(self, agents, exam_file, timeout=120):
        """ให้หลาย agent สอบข้อสอบเดียวกัน เทียบผล

        Args:
            agents: list of agent names
            exam_file: path to exam file
        """
        print(f"\n{'='*60}")
        print(f"  COMPARISON EXAM — {len(agents)} agents")
        print(f"{'='*60}\n")

        results = {}
        for agent in agents:
            print(f"\n--- Testing: {agent} ---")
            result = self.exam(agent, exam_file=exam_file, timeout=timeout)
            if result:
                results[agent] = result

        # สรุปเปรียบเทียบ
        if results:
            print(f"\n{'='*60}")
            print(f"  COMPARISON RESULTS")
            print(f"{'='*60}")
            print(f"  {'Agent':<30} {'Score':<15} {'Grade':<10}")
            print(f"  {'-'*30} {'-'*15} {'-'*10}")

            sorted_results = sorted(results.items(), key=lambda x: x[1]["percentage"], reverse=True)
            for agent, r in sorted_results:
                print(f"  {agent:<30} {r['score']}/{r['total']} ({r['percentage']:.1f}%) {r['grade']:<10}")

            winner = sorted_results[0]
            print(f"\n  Winner: {winner[0]} ({winner[1]['percentage']:.1f}%)")

        return results

    # ════════════════════════════════════════════════════════════════
    #  6. Training Pipeline — เทรนครบวงจร
    # ════════════════════════════════════════════════════════════════

    def full_training(self, agent_name, topics, teacher=None,
                      examples_per_topic=5, exam_questions=10, timeout=180):
        """เทรนครบวงจร: สร้าง dataset → เทรน → สอบ → รายงานผล

        Args:
            agent_name: ชื่อ agent ที่จะเทรน
            topics: list of หัวข้อที่อยากให้เก่ง
            teacher: model ครู
            examples_per_topic: ตัวอย่างต่อหัวข้อ
            exam_questions: จำนวนข้อสอบ
        """
        if not teacher:
            teacher = "qwen3:14b"

        dataset_name = f"training_{agent_name}"

        print(f"\n{'='*60}")
        print(f"  FULL TRAINING PIPELINE")
        print(f"  Agent: {agent_name}")
        print(f"  Teacher: {teacher}")
        print(f"  Topics: {len(topics)}")
        print(f"{'='*60}\n")

        # Step 1: สอบก่อนเทรน (pre-test)
        print("  STEP 1: Pre-training exam...")
        pre_exam = None
        exam_topic = ", ".join(topics[:3])
        exam_path = self.auto_generate_exam(exam_topic, exam_questions, teacher, timeout)
        if exam_path:
            pre_exam = self.exam(agent_name, exam_file=str(exam_path), timeout=timeout)

        # Step 2: สร้าง dataset
        print("\n  STEP 2: Creating training dataset...")
        self.distill(dataset_name, teacher, topics, examples_per_topic, timeout)

        # Step 3: เทรน
        print("\n  STEP 3: Training agent...")
        self.train(agent_name, dataset_name)

        # Step 4: สอบหลังเทรน (post-test)
        print("\n  STEP 4: Post-training exam...")
        post_exam = None
        if exam_path:
            post_exam = self.exam(agent_name, exam_file=str(exam_path), timeout=timeout)

        # สรุป
        print(f"\n{'='*60}")
        print(f"  TRAINING SUMMARY: {agent_name}")
        print(f"{'='*60}")
        if pre_exam:
            print(f"  Before: {pre_exam['percentage']:.1f}% ({pre_exam['grade']})")
        if post_exam:
            print(f"  After:  {post_exam['percentage']:.1f}% ({post_exam['grade']})")
        if pre_exam and post_exam:
            diff = post_exam['percentage'] - pre_exam['percentage']
            emoji_dir = "improved" if diff > 0 else "declined" if diff < 0 else "no change"
            print(f"  Change: {diff:+.1f}% ({emoji_dir})")
        print(f"{'='*60}\n")

    # ════════════════════════════════════════════════════════════════
    #  Internal helpers
    # ════════════════════════════════════════════════════════════════

    def _ask_model(self, model, prompt, timeout=120):
        """ส่ง prompt ให้ model"""
        try:
            result = subprocess.run(
                ["ollama", "run", model],
                input=prompt,
                capture_output=True, text=True,
                timeout=timeout,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except subprocess.TimeoutExpired:
            return None

    def _judge_with_keywords(self, response, expected, keywords, max_points):
        """ให้คะแนนด้วย keyword matching"""
        response_lower = response.lower()
        score = 0

        if keywords:
            matched = sum(1 for kw in keywords if kw.lower() in response_lower)
            score = int(max_points * (matched / len(keywords)))
        elif expected:
            # เทียบกับ expected answer
            expected_words = set(expected.lower().split())
            response_words = set(response_lower.split())
            if expected_words:
                overlap = len(expected_words & response_words) / len(expected_words)
                score = int(max_points * min(overlap * 1.5, 1.0))
        else:
            # ถ้าตอบมา ให้คะแนนขั้นต่ำ
            score = max_points // 2 if len(response) > 20 else 0

        return min(score, max_points)

    def _judge_with_ai(self, judge_model, question, expected, response,
                       max_points, timeout=120):
        """ให้ AI model อื่นเป็นกรรมการให้คะแนน"""
        prompt = f"""ให้คะแนนคำตอบนี้ (0-{max_points} คะแนน):

คำถาม: {question}

คำตอบที่คาดหวัง: {expected if expected else 'ไม่ระบุ'}

คำตอบที่ได้: {response[:1000]}

ให้คะแนนเป็นตัวเลขอย่างเดียว ไม่ต้องอธิบาย
ตัวอย่าง: 8"""

        result = self._ask_model(judge_model, prompt, timeout)
        if result:
            # Extract number
            numbers = re.findall(r'\d+', result)
            if numbers:
                score = int(numbers[0])
                return min(score, max_points)
        return 0

    def _calculate_grade(self, percentage):
        """คำนวณเกรด"""
        if percentage >= 90:
            return "A (Excellent)"
        elif percentage >= 80:
            return "B+ (Very Good)"
        elif percentage >= 70:
            return "B (Good)"
        elif percentage >= 60:
            return "C+ (Fair)"
        elif percentage >= 50:
            return "C (Pass)"
        elif percentage >= 40:
            return "D (Poor)"
        else:
            return "F (Fail)"


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Training Room — ห้องฝึกอบรม AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # สร้าง dataset จากโฟลเดอร์ code
  python training_room.py dataset-from-code my-data ./src --type .py

  # Distillation — ให้ model ใหญ่สอน model เล็ก
  python training_room.py distill my-data --teacher qwen3:14b --topics "Python OOP" "FastAPI" "Testing"

  # เทรน agent
  python training_room.py train my-agent my-data

  # สร้างข้อสอบอัตโนมัติ
  python training_room.py gen-exam "Python basics" --questions 10

  # สอบ agent
  python training_room.py exam my-agent --exam-file python_basics.json

  # เทรนครบวงจร (สร้าง data + เทรน + สอบ)
  python training_room.py full my-agent --teacher qwen3:14b --topics "Unity C#" "Game Physics"

  # เปรียบเทียบหลาย agents
  python training_room.py compare --agents ai-coder1 ai-coder2 --exam python_basics.json
        """,
    )

    sub = parser.add_subparsers(dest="command")

    # dataset from code
    p_dfc = sub.add_parser("dataset-from-code", help="สร้าง dataset จากไฟล์ code")
    p_dfc.add_argument("name", help="ชื่อ dataset")
    p_dfc.add_argument("folder", help="โฟลเดอร์ที่มี code")
    p_dfc.add_argument("--type", default=".py", help="นามสกุลไฟล์ (default: .py)")

    # distill
    p_dist = sub.add_parser("distill", help="Distillation — ให้ model ใหญ่สร้าง data")
    p_dist.add_argument("name", help="ชื่อ dataset")
    p_dist.add_argument("--teacher", default="qwen3:14b", help="Model ครู")
    p_dist.add_argument("--topics", nargs="+", required=True, help="หัวข้อ")
    p_dist.add_argument("--examples", type=int, default=5, help="ตัวอย่างต่อหัวข้อ")

    # train
    p_train = sub.add_parser("train", help="เทรน agent")
    p_train.add_argument("agent", help="ชื่อ agent")
    p_train.add_argument("dataset", help="ชื่อ dataset")

    # gen-exam
    p_gexam = sub.add_parser("gen-exam", help="สร้างข้อสอบอัตโนมัติ")
    p_gexam.add_argument("topic", help="หัวข้อ")
    p_gexam.add_argument("--questions", type=int, default=10)
    p_gexam.add_argument("--teacher", default="qwen3:8b")

    # exam
    p_exam = sub.add_parser("exam", help="สอบ agent")
    p_exam.add_argument("agent", help="ชื่อ agent")
    p_exam.add_argument("--exam-file", required=True, help="ไฟล์ข้อสอบ")
    p_exam.add_argument("--judge", help="Model ที่ให้คะแนน")

    # compare
    p_comp = sub.add_parser("compare", help="เปรียบเทียบหลาย agents")
    p_comp.add_argument("--agents", nargs="+", required=True)
    p_comp.add_argument("--exam", required=True, help="ไฟล์ข้อสอบ")

    # full training
    p_full = sub.add_parser("full", help="เทรนครบวงจร")
    p_full.add_argument("agent", help="ชื่อ agent")
    p_full.add_argument("--teacher", default="qwen3:14b")
    p_full.add_argument("--topics", nargs="+", required=True)
    p_full.add_argument("--examples", type=int, default=5)
    p_full.add_argument("--exam-questions", type=int, default=10)

    args = parser.parse_args()
    trainer = TrainingRoom()

    if args.command == "dataset-from-code":
        trainer.create_dataset_from_files(args.name, args.folder, args.type)
    elif args.command == "distill":
        trainer.distill(args.name, args.teacher, args.topics, args.examples)
    elif args.command == "train":
        trainer.train(args.agent, args.dataset)
    elif args.command == "gen-exam":
        trainer.auto_generate_exam(args.topic, args.questions, args.teacher)
    elif args.command == "exam":
        trainer.exam(args.agent, exam_file=args.exam_file, judge=args.judge)
    elif args.command == "compare":
        trainer.compare(args.agents, args.exam)
    elif args.command == "full":
        trainer.full_training(args.agent, args.topics, args.teacher, args.examples, args.exam_questions)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
