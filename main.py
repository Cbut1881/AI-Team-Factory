"""AI Team Factory — โรงงานผลิตทีม AI ครบวงจร

เครื่องมือ 3 ส่วน:
  1. Factory  — ปั๊ม agent ใหม่ / สร้างทีม
  2. Runner   — สั่งทีมทำงาน (parallel, pipeline, debate)
  3. Training — ห้องฝึกอบรม (distill, train, exam, compare)

Usage:
  python main.py factory create game-coder "เขียน game logic"
  python main.py factory team game-studio --config teams/game_studio.json
  python main.py factory list

  python main.py run ask game-coder "สร้างระบบ inventory ด้วย Python"
  python main.py run pipeline "สร้างเกม snake" --team game-studio
  python main.py run parallel "วิเคราะห์ปัญหา performance" --team game-studio

  python main.py train distill coder-data --teacher qwen3:14b --topics "Python" "FastAPI"
  python main.py train train game-coder coder-data
  python main.py train exam game-coder --exam-file python_basics.json
  python main.py train full game-coder --teacher qwen3:14b --topics "Pygame" "Game Design"
"""

import sys


def print_banner():
    print("""
    ╔══════════════════════════════════════════════════╗
    ║           AI TEAM FACTORY v1.0                   ║
    ║      โรงงานผลิตทีม AI ครบวงจร                      ║
    ╠══════════════════════════════════════════════════╣
    ║                                                  ║
    ║  factory  — ปั๊ม agent / สร้างทีม                  ║
    ║  run      — สั่งทีมทำงาน                           ║
    ║  train    — ห้องฝึกอบรม                            ║
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """)


def main():
    if len(sys.argv) < 2:
        print_banner()
        print("Usage: python main.py <module> <command> [args...]")
        print()
        print("Modules:")
        print("  factory  — สร้าง/จัดการ agents และทีม")
        print("  run      — สั่ง agents ทำงาน")
        print("  train    — ฝึกอบรม agents")
        print()
        print("Examples:")
        print("  python main.py factory create my-coder \"เขียน Python, FastAPI\"")
        print("  python main.py factory team my-team --config teams/web_agency.json")
        print("  python main.py factory list")
        print()
        print("  python main.py run ask my-coder \"สร้าง REST API\"")
        print("  python main.py run pipeline \"สร้างเว็บ\" --team my-team")
        print("  python main.py run debate \"React vs Vue\" --agents ai-a ai-b")
        print()
        print("  python main.py train full my-coder --topics \"Python\" \"Testing\"")
        print("  python main.py train exam my-coder --exam-file python.json")
        sys.exit(0)

    module = sys.argv[1]
    # Remove module name from argv so sub-module sees correct args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if module == "factory":
        from agent_factory import main as factory_main
        factory_main()
    elif module == "run":
        from team_runner import main as runner_main
        runner_main()
    elif module == "train":
        from training_room import main as training_main
        training_main()
    else:
        print(f"Unknown module: {module}")
        print("Available: factory, run, train")
        sys.exit(1)


if __name__ == "__main__":
    main()
