# AI Team Factory v1.0

โรงงานผลิตทีม AI ครบวงจร — สร้าง, จัดการ, สั่งงาน, ฝึกอบรม AI agents บน **Ollama** (local LLM)

```
╔══════════════════════════════════════════════════╗
║           AI TEAM FACTORY v1.0                   ║
║      โรงงานผลิตทีม AI ครบวงจร                      ║
╠══════════════════════════════════════════════════╣
║  factory  — ปั๊ม agent / สร้างทีม                  ║
║  run      — สั่งทีมทำงาน                           ║
║  train    — ห้องฝึกอบรม                            ║
╚══════════════════════════════════════════════════╝
```

## Features

### Factory — สร้าง Agent & ทีม
- สร้าง AI agent เฉพาะทางจาก base model ด้วยคำสั่งเดียว
- มี 7 role presets สำเร็จรูป: `coder`, `tester`, `planner`, `reviewer`, `designer`, `writer`, `researcher`
- สร้างทีมหลายตัวพร้อมกันจาก JSON config
- Auto-detect preset จากชื่อและความสามารถ

### Runner — สั่งทีมทำงาน
- **ask** — ถาม agent ตัวเดียว
- **parallel** — ส่งงานเดียวกันให้ทุกตัวทำพร้อมกัน
- **pipeline** — ส่งงานต่อกันเป็นสายพาน (output → input)
- **debate** — ให้ agents ถกเถียง หาคำตอบที่ดีที่สุด

### Training — ห้องฝึกอบรม
- **distill** — ให้ model ใหญ่ (teacher) สร้าง training data ให้ model เล็ก
- **train** — เทรน agent ด้วย dataset
- **exam** — สอบวัดผลอัตโนมัติ พร้อมให้เกรด
- **compare** — เปรียบเทียบผลสอบหลาย agents
- **full** — เทรนครบวงจร (สร้างข้อสอบ → pre-test → distill → train → post-test)

### Dashboard — Web UI
- Real-time dashboard ผ่าน Flask + Socket.IO
- สร้าง agent, สั่งงานทีม, ฝึกอบรม ผ่านหน้าเว็บ
- แสดงสถานะ GPU, Ollama, activity log

## Requirements

- **Python 3.8+**
- **[Ollama](https://ollama.com/)** — ติดตั้งและรันก่อนใช้งาน

## Installation

```bash
git clone https://github.com/Cbut1881/AI-Team-Factory.git
cd AI-Team-Factory
pip install -r requirements.txt
```

## Quick Start

### 1. สร้าง Agent

```bash
# สร้าง agent ตัวเดียว
python main.py factory create game-coder "เขียน game logic, Pygame, physics"

# สร้างพร้อมระบุ preset
python main.py factory create api-tester "ทดสอบ REST API" --preset tester

# ดู agents ทั้งหมด
python main.py factory list

# ดู presets ที่ใช้ได้
python main.py factory presets
```

### 2. สร้างทีม

```bash
# สร้างทีมจาก config file
python main.py factory team game-studio --config teams/game_studio.json

# ดูทีมทั้งหมด
python main.py factory teams
```

### 3. สั่งทีมทำงาน

```bash
# ถาม agent ตัวเดียว
python main.py run ask game-coder "สร้างระบบ inventory ด้วย Python"

# ส่งงานให้ทุกตัวทำพร้อมกัน
python main.py run parallel "วิเคราะห์ปัญหา performance" --team game-studio

# ส่งงานต่อกันเป็นสายพาน
python main.py run pipeline "สร้างเกม snake" --team game-studio

# ให้ agents ถกเถียงกัน
python main.py run debate "React vs Vue" --agents ai-web-frontend ai-web-backend
```

### 4. ฝึกอบรม

```bash
# เทรนครบวงจร
python main.py train full game-coder --teacher qwen3:14b --topics "Pygame" "Game Design"

# สร้าง dataset จาก AI ตัวเก่ง
python main.py train distill coder-data --teacher qwen3:14b --topics "Python" "FastAPI"

# เทรน agent ด้วย dataset
python main.py train train game-coder coder-data

# สอบวัดผล
python main.py train exam game-coder --exam-file python_basics.json

# สร้างข้อสอบอัตโนมัติ
python main.py train gen-exam "Python basics" --questions 10
```

### 5. Dashboard

```bash
python dashboard.py
# เปิด http://localhost:5555
```

## Team Configs

มี team config สำเร็จรูป 3 ทีม:

| ไฟล์ | ทีม | สมาชิก |
|------|-----|--------|
| `teams/game_studio.json` | Game Studio | designer, coder, tester, writer |
| `teams/web_agency.json` | Web Agency | planner, frontend, backend, tester |
| `teams/product_company.json` | Product Company | researcher, designer, developer, qa, marketer |

## Base Models

| Model | ขนาด | จุดเด่น | เหมาะกับ |
|-------|------|---------|----------|
| `qwen3:8b` | 5 GB | reasoning, multilingual, Thai | planner, manager, designer |
| `deepseek-coder-v2:16b` | 9 GB | code generation, debugging | coder, developer |
| `qwen3:4b` | 2.5 GB | fast, lightweight | tester, helper |
| `qwen3:14b` | 9 GB | strong reasoning | architect, reviewer |
| `gemma3:12b` | 8 GB | creative, multimodal | creative, storyteller |

## Project Structure

```
AI-Team-Factory/
├── main.py              # CLI entry point
├── agent_factory.py     # สร้าง/จัดการ agents & teams
├── team_runner.py       # สั่งงาน: ask, parallel, pipeline, debate
├── training_room.py     # ฝึกอบรม: distill, train, exam
├── dashboard.py         # Web dashboard (port 5555)
├── requirements.txt
├── templates/
│   └── index.html       # Dashboard frontend
└── teams/               # Team config files
    ├── game_studio.json
    ├── web_agency.json
    └── product_company.json
```

## License

MIT
