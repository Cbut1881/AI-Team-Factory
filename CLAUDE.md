# CLAUDE.md — AI Team Factory

## Project Overview
AI Team Factory v1.0 — โรงงานผลิตทีม AI ครบวงจร ใช้ Ollama เป็น backend สำหรับสร้าง/จัดการ/ฝึกอบรม AI agents บน local LLM

## Tech Stack
- **Language**: Python 3
- **Backend**: Ollama (local LLM inference)
- **Web**: Flask + Flask-SocketIO (threading mode)
- **Frontend**: Single-page HTML (`templates/index.html`)
- **Dependencies**: `pip install -r requirements.txt` (flask, flask-socketio, simple-websocket)

## Project Structure
```
AI-Team-Factory/
├── main.py              # CLI entry point — routes to factory/run/train
├── agent_factory.py     # AgentFactory — สร้าง/จัดการ agents & teams
├── team_runner.py       # TeamRunner — ask/parallel/pipeline/debate
├── training_room.py     # TrainingRoom — distill/train/exam/compare
├── dashboard.py         # Flask+SocketIO dashboard (port 5555)
├── requirements.txt
├── templates/
│   └── index.html       # Dashboard frontend
├── teams/               # Team config JSON files
│   ├── game_studio.json
│   ├── web_agency.json
│   └── product_company.json
├── agents/              # Generated .Modelfile files (runtime)
├── datasets/            # Training .jsonl files (runtime)
├── exams/               # Exam .json files (runtime)
├── results/             # Runner output (runtime)
├── training_results/    # Training/exam results (runtime)
└── registry.json        # Central registry of agents & teams (runtime)
```

## How to Run
```bash
# CLI
python main.py factory create my-coder "เขียน Python"
python main.py factory team my-team --config teams/game_studio.json
python main.py run ask my-coder "สร้าง REST API"
python main.py run pipeline "สร้างเว็บ" --team my-team
python main.py train full my-coder --topics "Python" "Testing"

# Dashboard
python dashboard.py   # → http://localhost:5555
```

## Code Conventions
- ภาษาไทยใน comments, docstrings, CLI output, UI text
- Agent names ต้อง prefix ด้วย `ai-` เสมอ (เช่น `ai-game-coder`)
- Ollama interaction ผ่าน `subprocess.run(["ollama", ...])`
- Registry เป็น JSON file กลาง มี dict `agents` และ `teams`
- Dashboard ใช้ background threads + Socket.IO events สำหรับ real-time updates
- Event names: `progress`, `stream`, `run_event`, `train_event`, `activity`

## Architecture Notes
- Training ใช้ "prompt-injection" — ยัด knowledge เข้า system prompt แล้ว rebuild Modelfile (ไม่ใช่ fine-tune จริง)
- Runner modes: ask (single), parallel (concurrent), pipeline (chain), debate (multi-round)
- Full training pipeline: gen exam → pre-test → distill → train → post-test → compare

## Important Constants
- Dashboard port: **5555**
- Default timeout: 120s (runner), 180s (training)
- Role presets: coder, tester, planner, reviewer, designer, writer, researcher
- Base models: qwen3:8b, deepseek-coder-v2:16b, qwen3:4b, qwen3:14b, gemma3:12b

## Do NOT
- อย่าแก้ไข `registry.json` โดยตรง — ใช้ AgentFactory methods
- อย่าสร้างไฟล์ใน `agents/`, `datasets/`, `results/`, `training_results/` มือ — มันเป็น runtime output
- อย่าเปลี่ยน agent name format (ต้อง `ai-` prefix เสมอ)
- อย่า hardcode model names — ใช้ BASE_MODELS dict หรือ ROLE_PRESETS
