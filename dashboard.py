"""AI Team Factory — Dashboard Server
Flask + Socket.IO real-time dashboard for managing AI agents, teams, and training.

Usage:
    python dashboard.py
    # Open http://localhost:5555
"""

import json
import os
import subprocess
import sys
import threading
import time
import re
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# ── Imports from factory modules ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from agent_factory import AgentFactory, ROLE_PRESETS, BASE_MODELS, REGISTRY_FILE, AGENTS_DIR
from team_runner import TeamRunner, RESULTS_DIR
from training_room import TrainingRoom, DATASETS_DIR, EXAMS_DIR
from training_room import RESULTS_DIR as TRAIN_RESULTS_DIR

# ── Flask App ──────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "ai-team-factory-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Shared instances ───────────────────────────────────────────────
factory = AgentFactory()
runner = TeamRunner()
trainer = TrainingRoom()

# Activity log (in-memory ring buffer)
activity_log = []
MAX_ACTIVITY = 200


def add_activity(action, detail, status="info"):
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "action": action,
        "detail": detail,
        "status": status,
    }
    activity_log.insert(0, entry)
    if len(activity_log) > MAX_ACTIVITY:
        activity_log.pop()
    socketio.emit("activity", entry)


def reload_registry():
    """Reload registry from disk for all instances."""
    factory.registry = factory._load_registry()
    runner.registry = runner._load_registry()


# ══════════════════════════════════════════════════════════════════
#  Pages
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ══════════════════════════════════════════════════════════════════
#  API — Agents & Teams
# ══════════════════════════════════════════════════════════════════

@app.route("/api/agents")
def api_agents():
    reload_registry()
    return jsonify(factory.registry.get("agents", {}))


@app.route("/api/teams")
def api_teams():
    reload_registry()
    return jsonify(factory.registry.get("teams", {}))


@app.route("/api/presets")
def api_presets():
    data = {}
    for name, cfg in ROLE_PRESETS.items():
        data[name] = {
            "base_model": cfg["base_model"],
            "temperature": cfg["temperature"],
            "context_length": cfg["context_length"],
        }
    return jsonify(data)


@app.route("/api/base-models")
def api_base_models():
    return jsonify(BASE_MODELS)


# ══════════════════════════════════════════════════════════════════
#  API — Factory
# ══════════════════════════════════════════════════════════════════

@app.route("/api/factory/create", methods=["POST"])
def api_factory_create():
    data = request.json or {}
    name = data.get("name", "").strip()
    expertise = data.get("expertise", "").strip()
    if not name or not expertise:
        return jsonify({"error": "name and expertise required"}), 400

    def do_create():
        socketio.emit("progress", {
            "stage": "create_agent",
            "agent": name,
            "message": f"กำลังสร้าง agent: {name}...",
            "percent": 10,
        })
        add_activity("สร้าง Agent", f"เริ่มสร้าง {name}", "running")

        success = factory.create(
            name=name,
            expertise=expertise,
            preset=data.get("preset"),
            base_model=data.get("base_model"),
            temperature=data.get("temperature"),
            system_prompt=data.get("system_prompt"),
        )

        socketio.emit("progress", {
            "stage": "create_agent",
            "agent": name,
            "message": f"Agent {name} {'สร้างสำเร็จ!' if success else 'สร้างไม่สำเร็จ'}",
            "percent": 100,
            "done": True,
            "success": success,
        })
        status = "success" if success else "error"
        add_activity("สร้าง Agent", f"{'สำเร็จ' if success else 'ล้มเหลว'}: {name}", status)

    threading.Thread(target=do_create, daemon=True).start()
    return jsonify({"status": "started", "agent": name})


@app.route("/api/factory/team", methods=["POST"])
def api_factory_team():
    data = request.json or {}
    team_name = data.get("team_name", "").strip()
    members = data.get("members", [])
    if not team_name or not members:
        return jsonify({"error": "team_name and members required"}), 400

    def do_create_team():
        total = len(members)
        add_activity("สร้างทีม", f"เริ่มสร้างทีม {team_name} ({total} สมาชิก)", "running")

        for i, member in enumerate(members, 1):
            socketio.emit("progress", {
                "stage": "create_team",
                "team": team_name,
                "agent": member.get("name", ""),
                "message": f"[{i}/{total}] กำลังสร้าง {member.get('name', '')}...",
                "percent": int(i / total * 100),
                "current": i,
                "total": total,
            })

            factory.create(
                name=member["name"],
                expertise=member.get("expertise", ""),
                preset=member.get("preset"),
                base_model=member.get("base_model"),
                temperature=member.get("temperature"),
                system_prompt=member.get("system_prompt"),
            )

        # Register team
        factory.registry["teams"][team_name] = {
            "members": [f"ai-{m['name']}" for m in members],
            "created": datetime.now().isoformat(),
        }
        factory._save_registry()

        socketio.emit("progress", {
            "stage": "create_team",
            "team": team_name,
            "message": f"ทีม {team_name} สร้างเสร็จแล้ว!",
            "percent": 100,
            "done": True,
            "success": True,
        })
        add_activity("สร้างทีม", f"สำเร็จ: {team_name}", "success")

    threading.Thread(target=do_create_team, daemon=True).start()
    return jsonify({"status": "started", "team": team_name})


@app.route("/api/factory/agent/<name>", methods=["DELETE"])
def api_factory_delete(name):
    factory.delete_agent(name)
    add_activity("ลบ Agent", f"ลบ {name}", "warning")
    return jsonify({"status": "deleted", "agent": name})


# ══════════════════════════════════════════════════════════════════
#  API — Runner
# ══════════════════════════════════════════════════════════════════

@app.route("/api/run/ask", methods=["POST"])
def api_run_ask():
    data = request.json or {}
    agent = data.get("agent", "").strip()
    task = data.get("task", "").strip()
    if not agent or not task:
        return jsonify({"error": "agent and task required"}), 400

    def do_ask():
        full_name = agent if agent.startswith("ai-") else f"ai-{agent}"
        add_activity("ถาม Agent", f"{full_name}: {task[:60]}...", "running")
        socketio.emit("run_event", {
            "mode": "ask",
            "agent": full_name,
            "status": "running",
            "message": f"กำลังถาม {full_name}...",
        })

        # Stream response using subprocess with line-by-line output
        try:
            proc = subprocess.Popen(
                ["ollama", "run", full_name],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            proc.stdin.write(task)
            proc.stdin.close()

            full_response = []
            for line in iter(proc.stdout.readline, ""):
                full_response.append(line)
                socketio.emit("stream", {
                    "mode": "ask",
                    "agent": full_name,
                    "chunk": line,
                    "done": False,
                })
            proc.wait(timeout=300)

            response_text = "".join(full_response)
            socketio.emit("stream", {
                "mode": "ask",
                "agent": full_name,
                "chunk": "",
                "response": response_text,
                "done": True,
            })
            add_activity("ถาม Agent", f"{full_name} ตอบแล้ว ({len(response_text)} chars)", "success")
        except Exception as e:
            socketio.emit("stream", {
                "mode": "ask",
                "agent": full_name,
                "chunk": "",
                "error": str(e),
                "done": True,
            })
            add_activity("ถาม Agent", f"{full_name} error: {e}", "error")

    threading.Thread(target=do_ask, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/run/pipeline", methods=["POST"])
def api_run_pipeline():
    data = request.json or {}
    task = data.get("task", "").strip()
    agents = data.get("agents", [])
    team_name = data.get("team")
    if not task:
        return jsonify({"error": "task required"}), 400

    def do_pipeline():
        reload_registry()
        agent_list = agents
        if team_name and not agent_list:
            team = runner.registry["teams"].get(team_name)
            if team:
                agent_list = team["members"]

        if not agent_list:
            socketio.emit("run_event", {"mode": "pipeline", "error": "No agents"})
            return

        total = len(agent_list)
        add_activity("Pipeline", f"เริ่ม pipeline {total} stages", "running")
        current_input = task

        for i, agent in enumerate(agent_list, 1):
            socketio.emit("run_event", {
                "mode": "pipeline",
                "stage": i,
                "total": total,
                "agent": agent,
                "status": "running",
                "message": f"Stage {i}/{total}: {agent}",
                "input": current_input[:500],
            })

            expertise = runner.registry["agents"].get(agent, {}).get("expertise", "")
            if i == 1:
                prompt = task
            else:
                prev_agent = agent_list[i - 2]
                prompt = (
                    f"Original task: {task}\n\n"
                    f"Previous agent ({prev_agent}) output:\n"
                    f"---\n{current_input}\n---\n\n"
                    f"Your role: {expertise}\n"
                    f"Continue the work based on the above. Do your part."
                )

            result = runner._ask_agent(agent, prompt, 180)

            socketio.emit("run_event", {
                "mode": "pipeline",
                "stage": i,
                "total": total,
                "agent": agent,
                "status": "done" if not result["error"] else "error",
                "response": result["response"],
                "error": result.get("error"),
            })

            if result["error"]:
                break
            current_input = result["response"]

        socketio.emit("run_event", {"mode": "pipeline", "status": "complete"})
        add_activity("Pipeline", f"เสร็จสิ้น {total} stages", "success")

    threading.Thread(target=do_pipeline, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/run/parallel", methods=["POST"])
def api_run_parallel():
    data = request.json or {}
    task = data.get("task", "").strip()
    agents = data.get("agents", [])
    team_name = data.get("team")
    if not task:
        return jsonify({"error": "task required"}), 400

    def do_parallel():
        reload_registry()
        agent_list = agents
        if team_name and not agent_list:
            team = runner.registry["teams"].get(team_name)
            if team:
                agent_list = team["members"]

        if not agent_list:
            socketio.emit("run_event", {"mode": "parallel", "error": "No agents"})
            return

        add_activity("Parallel", f"เริ่ม parallel {len(agent_list)} agents", "running")

        for agent in agent_list:
            socketio.emit("run_event", {
                "mode": "parallel",
                "agent": agent,
                "status": "running",
            })

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(agent_list)) as executor:
            futures = {
                executor.submit(runner._ask_agent, agent, task, 180): agent
                for agent in agent_list
            }
            for future in as_completed(futures):
                agent = futures[future]
                result = future.result()
                socketio.emit("run_event", {
                    "mode": "parallel",
                    "agent": agent,
                    "status": "done" if not result["error"] else "error",
                    "response": result["response"],
                    "error": result.get("error"),
                })

        socketio.emit("run_event", {"mode": "parallel", "status": "complete"})
        add_activity("Parallel", f"เสร็จสิ้น {len(agent_list)} agents", "success")

    threading.Thread(target=do_parallel, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/run/debate", methods=["POST"])
def api_run_debate():
    data = request.json or {}
    question = data.get("question", "").strip()
    agents = data.get("agents", [])
    rounds = data.get("rounds", 2)
    if not question or len(agents) < 2:
        return jsonify({"error": "question and at least 2 agents required"}), 400

    def do_debate():
        add_activity("Debate", f"เริ่ม debate {len(agents)} agents, {rounds} rounds", "running")

        responses = {}

        # Round 1
        socketio.emit("run_event", {
            "mode": "debate",
            "round": 1,
            "total_rounds": rounds,
            "status": "round_start",
        })

        for agent in agents:
            socketio.emit("run_event", {
                "mode": "debate",
                "round": 1,
                "agent": agent,
                "status": "running",
            })
            result = runner._ask_agent(agent, question, 180)
            responses[agent] = result["response"]
            socketio.emit("run_event", {
                "mode": "debate",
                "round": 1,
                "agent": agent,
                "status": "done",
                "response": result["response"],
            })

        # Subsequent rounds
        for r in range(2, rounds + 1):
            socketio.emit("run_event", {
                "mode": "debate",
                "round": r,
                "total_rounds": rounds,
                "status": "round_start",
            })

            new_responses = {}
            for agent in agents:
                others = "\n\n".join(
                    f"{a}: {resp[:500]}"
                    for a, resp in responses.items()
                    if a != agent
                )
                prompt = (
                    f"Question: {question}\n\n"
                    f"Your previous answer: {responses[agent][:500]}\n\n"
                    f"Other agents said:\n{others}\n\n"
                    f"Reconsider your answer. Keep what's right, fix what's wrong. "
                    f"Give your final answer."
                )
                socketio.emit("run_event", {
                    "mode": "debate",
                    "round": r,
                    "agent": agent,
                    "status": "running",
                })
                result = runner._ask_agent(agent, prompt, 180)
                new_responses[agent] = result["response"]
                socketio.emit("run_event", {
                    "mode": "debate",
                    "round": r,
                    "agent": agent,
                    "status": "done",
                    "response": result["response"],
                })

            responses = new_responses

        socketio.emit("run_event", {"mode": "debate", "status": "complete"})
        add_activity("Debate", f"เสร็จสิ้น {rounds} rounds", "success")

    threading.Thread(target=do_debate, daemon=True).start()
    return jsonify({"status": "started"})


# ══════════════════════════════════════════════════════════════════
#  API — Training
# ══════════════════════════════════════════════════════════════════

@app.route("/api/train/distill", methods=["POST"])
def api_train_distill():
    data = request.json or {}
    dataset_name = data.get("dataset_name", "").strip()
    teacher = data.get("teacher", "qwen3:14b")
    topics = data.get("topics", [])
    examples_per_topic = data.get("examples_per_topic", 5)
    if not dataset_name or not topics:
        return jsonify({"error": "dataset_name and topics required"}), 400

    def do_distill():
        add_activity("Distill", f"เริ่ม distill {dataset_name}", "running")
        total_topics = len(topics)

        for i, topic in enumerate(topics, 1):
            socketio.emit("train_event", {
                "stage": "distill",
                "topic": topic,
                "current": i,
                "total": total_topics,
                "percent": int(i / total_topics * 100),
                "message": f"[{i}/{total_topics}] Topic: {topic}",
            })

        # Run actual distillation
        examples = trainer.distill(dataset_name, teacher, topics, examples_per_topic)

        socketio.emit("train_event", {
            "stage": "distill",
            "status": "complete",
            "message": f"Distillation เสร็จสิ้น: {len(examples)} examples",
            "examples_count": len(examples),
        })
        add_activity("Distill", f"สำเร็จ: {len(examples)} examples", "success")

    threading.Thread(target=do_distill, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/train/train", methods=["POST"])
def api_train_train():
    data = request.json or {}
    agent_name = data.get("agent_name", "").strip()
    dataset_name = data.get("dataset_name", "").strip()
    if not agent_name or not dataset_name:
        return jsonify({"error": "agent_name and dataset_name required"}), 400

    def do_train():
        add_activity("Train", f"เริ่มเทรน {agent_name} ด้วย {dataset_name}", "running")
        socketio.emit("train_event", {
            "stage": "train",
            "agent": agent_name,
            "dataset": dataset_name,
            "message": f"กำลังเทรน {agent_name}...",
            "percent": 30,
        })

        success = trainer.train(agent_name, dataset_name)

        socketio.emit("train_event", {
            "stage": "train",
            "agent": agent_name,
            "status": "complete",
            "success": success,
            "message": f"{'เทรนสำเร็จ!' if success else 'เทรนไม่สำเร็จ'}",
            "percent": 100,
        })
        add_activity("Train", f"{'สำเร็จ' if success else 'ล้มเหลว'}: {agent_name}", "success" if success else "error")

    threading.Thread(target=do_train, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/train/exam", methods=["POST"])
def api_train_exam():
    data = request.json or {}
    agent_name = data.get("agent_name", "").strip()
    exam_file = data.get("exam_file", "").strip()
    judge = data.get("judge")
    if not agent_name or not exam_file:
        return jsonify({"error": "agent_name and exam_file required"}), 400

    def do_exam():
        add_activity("Exam", f"เริ่มสอบ {agent_name}", "running")

        # Load exam questions
        exam_path = Path(exam_file)
        if not exam_path.exists():
            exam_path = EXAMS_DIR / exam_file
        if not exam_path.exists():
            socketio.emit("train_event", {
                "stage": "exam",
                "status": "error",
                "message": f"ไม่พบไฟล์ข้อสอบ: {exam_file}",
            })
            return

        exam_data = json.loads(exam_path.read_text(encoding="utf-8"))
        questions = exam_data.get("questions", [])
        total_q = len(questions)

        full_name = f"ai-{agent_name}" if not agent_name.startswith("ai-") else agent_name
        total_points = 0
        earned_points = 0
        results = []

        for i, q in enumerate(questions, 1):
            socketio.emit("train_event", {
                "stage": "exam",
                "question_num": i,
                "total_questions": total_q,
                "question": q["question"][:100],
                "percent": int(i / total_q * 100),
                "message": f"Q{i}/{total_q}: {q['question'][:60]}...",
            })

            pts = q.get("points", 10)
            total_points += pts

            response = trainer._ask_model(full_name, q["question"], 120)
            if not response:
                score = 0
            elif judge:
                score = trainer._judge_with_ai(judge, q["question"], q.get("expected", ""), response, pts)
            else:
                score = trainer._judge_with_keywords(response, q.get("expected", ""), q.get("keywords", []), pts)

            earned_points += score
            results.append({
                "question": q["question"],
                "response": (response or "")[:500],
                "score": score,
                "max": pts,
                "passed": score >= pts * 0.6,
            })

            socketio.emit("train_event", {
                "stage": "exam_result",
                "question_num": i,
                "score": score,
                "max": pts,
                "passed": score >= pts * 0.6,
            })

        percentage = (earned_points / total_points * 100) if total_points > 0 else 0
        grade = trainer._calculate_grade(percentage)

        socketio.emit("train_event", {
            "stage": "exam",
            "status": "complete",
            "agent": full_name,
            "score": earned_points,
            "total": total_points,
            "percentage": round(percentage, 1),
            "grade": grade,
            "results": results,
        })
        add_activity("Exam", f"{full_name}: {percentage:.1f}% ({grade})", "success")

    threading.Thread(target=do_exam, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/train/full", methods=["POST"])
def api_train_full():
    data = request.json or {}
    agent_name = data.get("agent_name", "").strip()
    topics = data.get("topics", [])
    teacher = data.get("teacher", "qwen3:14b")
    examples_per_topic = data.get("examples_per_topic", 5)
    exam_questions = data.get("exam_questions", 10)
    if not agent_name or not topics:
        return jsonify({"error": "agent_name and topics required"}), 400

    def do_full():
        add_activity("Full Training", f"เริ่ม full training {agent_name}", "running")

        steps = [
            "สร้างข้อสอบ (Pre-test)",
            "สอบก่อนเทรน",
            "สร้าง Dataset (Distillation)",
            "เทรน Agent",
            "สอบหลังเทรน",
        ]

        # Step 1: Generate exam
        socketio.emit("train_event", {
            "stage": "full",
            "step": 1,
            "total_steps": 5,
            "step_name": steps[0],
            "percent": 10,
            "message": "กำลังสร้างข้อสอบ...",
        })
        exam_topic = ", ".join(topics[:3])
        exam_path = trainer.auto_generate_exam(exam_topic, exam_questions, teacher, 180)

        # Step 2: Pre-test
        socketio.emit("train_event", {
            "stage": "full",
            "step": 2,
            "total_steps": 5,
            "step_name": steps[1],
            "percent": 25,
            "message": "กำลังสอบก่อนเทรน...",
        })
        pre_exam = None
        if exam_path:
            pre_exam = trainer.exam(agent_name, exam_file=str(exam_path), timeout=120)
            if pre_exam:
                socketio.emit("train_event", {
                    "stage": "full_pre_exam",
                    "score": pre_exam["score"],
                    "total": pre_exam["total"],
                    "percentage": pre_exam["percentage"],
                    "grade": pre_exam["grade"],
                })

        # Step 3: Distill
        dataset_name = f"training_{agent_name}"
        socketio.emit("train_event", {
            "stage": "full",
            "step": 3,
            "total_steps": 5,
            "step_name": steps[2],
            "percent": 45,
            "message": "กำลังสร้าง dataset...",
        })
        trainer.distill(dataset_name, teacher, topics, examples_per_topic, 180)

        # Step 4: Train
        socketio.emit("train_event", {
            "stage": "full",
            "step": 4,
            "total_steps": 5,
            "step_name": steps[3],
            "percent": 70,
            "message": "กำลังเทรน agent...",
        })
        trainer.train(agent_name, dataset_name)

        # Step 5: Post-test
        socketio.emit("train_event", {
            "stage": "full",
            "step": 5,
            "total_steps": 5,
            "step_name": steps[4],
            "percent": 90,
            "message": "กำลังสอบหลังเทรน...",
        })
        post_exam = None
        if exam_path:
            post_exam = trainer.exam(agent_name, exam_file=str(exam_path), timeout=120)
            if post_exam:
                socketio.emit("train_event", {
                    "stage": "full_post_exam",
                    "score": post_exam["score"],
                    "total": post_exam["total"],
                    "percentage": post_exam["percentage"],
                    "grade": post_exam["grade"],
                })

        # Summary
        summary = {"agent": agent_name}
        if pre_exam:
            summary["pre"] = {"percentage": pre_exam["percentage"], "grade": pre_exam["grade"]}
        if post_exam:
            summary["post"] = {"percentage": post_exam["percentage"], "grade": post_exam["grade"]}
        if pre_exam and post_exam:
            summary["improvement"] = round(post_exam["percentage"] - pre_exam["percentage"], 1)

        socketio.emit("train_event", {
            "stage": "full",
            "status": "complete",
            "percent": 100,
            "summary": summary,
            "message": "Full training เสร็จสิ้น!",
        })
        add_activity("Full Training", f"เสร็จสิ้น: {agent_name}", "success")

    threading.Thread(target=do_full, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/train/datasets")
def api_train_datasets():
    datasets = []
    if DATASETS_DIR.exists():
        for f in DATASETS_DIR.glob("*.jsonl"):
            line_count = sum(1 for _ in open(f, encoding="utf-8"))
            datasets.append({
                "name": f.stem,
                "file": f.name,
                "examples": line_count,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            })
    return jsonify(datasets)


@app.route("/api/train/exams")
def api_train_exams():
    exams = []
    if EXAMS_DIR.exists():
        for f in EXAMS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                exams.append({
                    "name": f.stem,
                    "file": f.name,
                    "questions": len(data.get("questions", [])),
                    "created": data.get("created", ""),
                })
            except (json.JSONDecodeError, OSError):
                pass
    return jsonify(exams)


@app.route("/api/train/results")
def api_train_results():
    results = []
    if TRAIN_RESULTS_DIR.exists():
        for f in sorted(TRAIN_RESULTS_DIR.glob("*.json"), reverse=True)[:50]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                data["_file"] = f.name
                results.append(data)
            except (json.JSONDecodeError, OSError):
                pass
    return jsonify(results)


# ══════════════════════════════════════════════════════════════════
#  API — Models / System
# ══════════════════════════════════════════════════════════════════

@app.route("/api/models")
def api_models():
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        models = []
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n")[1:]:  # skip header
                parts = line.split()
                if parts:
                    models.append({
                        "name": parts[0],
                        "id": parts[1] if len(parts) > 1 else "",
                        "size": parts[2] + " " + parts[3] if len(parts) > 3 else "",
                        "modified": " ".join(parts[4:]) if len(parts) > 4 else "",
                    })
        return jsonify({"status": "ok", "models": models})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "models": []})


@app.route("/api/system")
def api_system():
    # Check Ollama
    ollama_ok = False
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        ollama_ok = r.returncode == 0
    except Exception:
        pass

    # Check GPU (nvidia-smi)
    gpu_info = None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,gpu_name,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            if len(parts) >= 4:
                gpu_info = {
                    "memory_used": int(parts[0]),
                    "memory_total": int(parts[1]),
                    "name": parts[2],
                    "temperature": int(parts[3]),
                }
    except Exception:
        pass

    return jsonify({
        "ollama": ollama_ok,
        "gpu": gpu_info,
    })


@app.route("/api/activity")
def api_activity():
    return jsonify(activity_log[:50])


# ══════════════════════════════════════════════════════════════════
#  Socket.IO events
# ══════════════════════════════════════════════════════════════════

@socketio.on("connect")
def handle_connect():
    emit("connected", {"status": "ok", "time": datetime.now().isoformat()})


@socketio.on("ping_server")
def handle_ping():
    emit("pong_server", {"time": datetime.now().isoformat()})


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════╗
    ║       AI TEAM FACTORY — DASHBOARD                ║
    ║       http://localhost:5555                       ║
    ╚══════════════════════════════════════════════════╝
    """)
    add_activity("System", "Dashboard เริ่มทำงาน", "success")
    socketio.run(app, host="0.0.0.0", port=5555, debug=True, allow_unsafe_werkzeug=True)
