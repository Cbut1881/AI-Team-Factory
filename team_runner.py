"""Team Runner — ให้ AI หลายตัวทำงานร่วมกัน

ใช้งาน:
    runner = TeamRunner()
    result = runner.run("game-studio", "สร้างเกม puzzle 2D ด้วย Pygame")

    # หรือ run แบบ pipeline (ส่งต่อผลงานกัน)
    result = runner.pipeline("game-studio", "สร้างเกม snake", [
        "ai-game-designer",   # ออกแบบก่อน
        "ai-game-coder",      # เขียน code จาก design
        "ai-game-tester",     # ทดสอบ code
    ])
"""

import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

FACTORY_DIR = Path(__file__).parent
REGISTRY_FILE = FACTORY_DIR / "registry.json"
RESULTS_DIR = FACTORY_DIR / "results"


class TeamRunner:
    """สั่งให้ AI ทำงานร่วมกัน — parallel หรือ pipeline"""

    def __init__(self):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self):
        if REGISTRY_FILE.exists():
            try:
                return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"agents": {}, "teams": {}}

    def _ask_agent(self, agent_name, prompt, timeout=120):
        """ส่งงานให้ agent ตัวเดียว"""
        try:
            result = subprocess.run(
                ["ollama", "run", agent_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "agent": agent_name,
                "prompt": prompt,
                "response": result.stdout.strip(),
                "error": result.stderr.strip() if result.returncode != 0 else None,
                "timestamp": datetime.now().isoformat(),
            }
        except subprocess.TimeoutExpired:
            return {
                "agent": agent_name,
                "prompt": prompt,
                "response": "",
                "error": f"Timeout after {timeout}s",
                "timestamp": datetime.now().isoformat(),
            }

    # ── Parallel ────────────────────────────────────────────────────

    def parallel(self, task, agents=None, team_name=None, timeout=120):
        """ส่งงานเดียวกันให้ทุกตัวทำพร้อมกัน แล้วรวมผล

        Args:
            task: งานที่ต้องทำ
            agents: list of agent names หรือ
            team_name: ใช้ทีมที่สร้างไว้
        """
        if team_name:
            team = self.registry["teams"].get(team_name)
            if not team:
                print(f"Team '{team_name}' not found.")
                return None
            agents = team["members"]

        if not agents:
            print("No agents specified.")
            return None

        print(f"\n{'='*60}")
        print(f"  PARALLEL RUN — {len(agents)} agents")
        print(f"  Task: {task[:80]}")
        print(f"{'='*60}\n")

        results = []
        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {}
            for agent in agents:
                future = executor.submit(self._ask_agent, agent, task, timeout)
                futures[future] = agent

            for future in as_completed(futures):
                agent = futures[future]
                result = future.result()
                results.append(result)
                status = "OK" if not result["error"] else "ERROR"
                print(f"  [{status}] {agent} — {len(result['response'])} chars")

        # Save results
        self._save_results("parallel", task, results)
        return results

    # ── Pipeline ────────────────────────────────────────────────────

    def pipeline(self, task, agents=None, team_name=None, timeout=120):
        """ส่งงานต่อกันเป็นสายพาน — output ตัวก่อนเป็น input ตัวถัดไป

        Args:
            task: งานเริ่มต้น
            agents: list of agent names ตามลำดับ
            team_name: ใช้ทีมที่สร้างไว้ (ใช้ตามลำดับ members)
        """
        if team_name:
            team = self.registry["teams"].get(team_name)
            if not team:
                print(f"Team '{team_name}' not found.")
                return None
            agents = team["members"]

        if not agents:
            print("No agents specified.")
            return None

        print(f"\n{'='*60}")
        print(f"  PIPELINE RUN — {len(agents)} stages")
        print(f"  Task: {task[:80]}")
        print(f"{'='*60}\n")

        current_input = task
        results = []

        for i, agent in enumerate(agents, 1):
            expertise = self.registry["agents"].get(agent, {}).get("expertise", "")
            print(f"  Stage {i}/{len(agents)}: {agent}")
            print(f"  Expertise: {expertise}")
            print(f"  Input: {current_input[:100]}...")
            print(f"  Working...", end="", flush=True)

            # Build prompt with context from previous stage
            if i == 1:
                prompt = task
            else:
                prev_agent = agents[i - 2]
                prompt = (
                    f"Original task: {task}\n\n"
                    f"Previous agent ({prev_agent}) output:\n"
                    f"---\n{current_input}\n---\n\n"
                    f"Your role: {expertise}\n"
                    f"Continue the work based on the above. Do your part."
                )

            result = self._ask_agent(agent, prompt, timeout)
            results.append(result)

            if result["error"]:
                print(f" ERROR: {result['error']}")
                break
            else:
                print(f" Done ({len(result['response'])} chars)")
                current_input = result["response"]

            print()

        # Save results
        self._save_results("pipeline", task, results)

        print(f"{'='*60}")
        print(f"  PIPELINE COMPLETE — {len(results)} stages")
        print(f"{'='*60}\n")

        return results

    # ── Single Agent ────────────────────────────────────────────────

    def ask(self, agent_name, task, timeout=120):
        """ถาม agent ตัวเดียว"""
        if not agent_name.startswith("ai-"):
            agent_name = f"ai-{agent_name}"

        print(f"  Asking {agent_name}...")
        result = self._ask_agent(agent_name, task, timeout)

        if result["error"]:
            print(f"  Error: {result['error']}")
        else:
            print(f"\n{result['response']}")

        return result

    # ── Debate ──────────────────────────────────────────────────────

    def debate(self, question, agents, rounds=2, timeout=120):
        """ให้ agents ถกเถียงกัน หาคำตอบที่ดีที่สุด

        Args:
            question: คำถาม/ปัญหา
            agents: list of 2+ agent names
            rounds: จำนวนรอบถกเถียง
        """
        print(f"\n{'='*60}")
        print(f"  DEBATE — {len(agents)} agents, {rounds} rounds")
        print(f"  Question: {question[:80]}")
        print(f"{'='*60}\n")

        all_responses = []

        # Round 1: ทุกคนตอบอิสระ
        print(f"  Round 1: Initial responses")
        responses = {}
        for agent in agents:
            result = self._ask_agent(agent, question, timeout)
            responses[agent] = result["response"]
            print(f"    {agent}: {result['response'][:100]}...")
        all_responses.append(responses)

        # Subsequent rounds: ดูคำตอบคนอื่น แล้วปรับ
        for r in range(2, rounds + 1):
            print(f"\n  Round {r}: Responses with context")
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
                result = self._ask_agent(agent, prompt, timeout)
                new_responses[agent] = result["response"]
                print(f"    {agent}: {result['response'][:100]}...")
            responses = new_responses
            all_responses.append(responses)

        # Save
        results = [{"round": i + 1, "responses": r} for i, r in enumerate(all_responses)]
        self._save_results("debate", question, results)

        return results

    # ── Save Results ────────────────────────────────────────────────

    def _save_results(self, mode, task, results):
        """บันทึกผลลัพธ์"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = RESULTS_DIR / f"{mode}_{timestamp}.json"
        data = {
            "mode": mode,
            "task": task,
            "results": results,
            "timestamp": datetime.now().isoformat(),
        }
        filename.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"\n  Results saved: {filename}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Team Runner — สั่ง AI ทำงานร่วมกัน")
    sub = parser.add_subparsers(dest="command")

    # ask single agent
    p_ask = sub.add_parser("ask", help="ถาม agent ตัวเดียว")
    p_ask.add_argument("agent", help="ชื่อ agent")
    p_ask.add_argument("task", help="งานที่ต้องทำ")
    p_ask.add_argument("--timeout", type=int, default=120)

    # parallel
    p_par = sub.add_parser("parallel", help="ส่งงานให้ทุกตัวทำพร้อมกัน")
    p_par.add_argument("task", help="งาน")
    p_par.add_argument("--team", help="ชื่อทีม")
    p_par.add_argument("--agents", nargs="+", help="ชื่อ agents")
    p_par.add_argument("--timeout", type=int, default=120)

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="ส่งงานต่อกันเป็นสายพาน")
    p_pipe.add_argument("task", help="งานเริ่มต้น")
    p_pipe.add_argument("--team", help="ชื่อทีม")
    p_pipe.add_argument("--agents", nargs="+", help="ชื่อ agents ตามลำดับ")
    p_pipe.add_argument("--timeout", type=int, default=120)

    # debate
    p_debate = sub.add_parser("debate", help="ให้ agents ถกเถียงกัน")
    p_debate.add_argument("question", help="คำถาม")
    p_debate.add_argument("--agents", nargs="+", required=True)
    p_debate.add_argument("--rounds", type=int, default=2)
    p_debate.add_argument("--timeout", type=int, default=120)

    args = parser.parse_args()
    runner = TeamRunner()

    if args.command == "ask":
        runner.ask(args.agent, args.task, args.timeout)
    elif args.command == "parallel":
        runner.parallel(args.task, agents=args.agents, team_name=args.team, timeout=args.timeout)
    elif args.command == "pipeline":
        runner.pipeline(args.task, agents=args.agents, team_name=args.team, timeout=args.timeout)
    elif args.command == "debate":
        runner.debate(args.question, args.agents, args.rounds, args.timeout)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
