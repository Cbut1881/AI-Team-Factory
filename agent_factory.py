"""AI Agent Factory — ปั๊ม AI Agent เฉพาะทางจาก base model

ใช้งาน:
    factory = AgentFactory()
    factory.create("game-designer", "ออกแบบเกม วางระบบ gameplay เขียน GDD")
    factory.create("3d-modeler", "สร้าง 3D model, UV mapping, texturing")
    factory.list_agents()
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────

FACTORY_DIR = Path(__file__).parent
AGENTS_DIR = FACTORY_DIR / "agents"
REGISTRY_FILE = FACTORY_DIR / "registry.json"

# Base models ที่ใช้ปั๊ม agent
BASE_MODELS = {
    "qwen3:8b": {
        "size": "5 GB",
        "strength": "reasoning, planning, multilingual, Thai",
        "best_for": ["planner", "manager", "analyst", "writer", "designer"],
    },
    "deepseek-coder-v2:16b": {
        "size": "9 GB",
        "strength": "code generation, debugging, code review",
        "best_for": ["coder", "developer", "debugger", "devops"],
    },
    "qwen3:4b": {
        "size": "2.5 GB",
        "strength": "fast, lightweight, good for simple tasks",
        "best_for": ["tester", "formatter", "helper", "translator"],
    },
    "qwen3:14b": {
        "size": "9 GB",
        "strength": "strong reasoning, complex tasks",
        "best_for": ["architect", "researcher", "lead", "reviewer"],
    },
    "gemma3:12b": {
        "size": "8 GB",
        "strength": "creative, multimodal understanding",
        "best_for": ["creative", "storyteller", "ui-designer", "content"],
    },
}

# ── Modelfile Template ──────────────────────────────────────────────

MODELFILE_TEMPLATE = """FROM {base_model}

PARAMETER temperature {temperature}
PARAMETER top_p {top_p}
PARAMETER num_ctx {context_length}

SYSTEM \"\"\"
# Role: {role_name}
# Expertise: {expertise}

{system_prompt}

## Rules
- ตอบตรงประเด็น ไม่อ้อมค้อม
- ถ้าไม่แน่ใจ บอกตรงๆ ว่าไม่แน่ใจ
- ให้ code ที่ run ได้จริง ไม่ใช่ pseudo code
- ใช้ภาษาไทยหรืออังกฤษตามที่ถูกถาม
\"\"\"
"""

# ── Role Templates ──────────────────────────────────────────────────

ROLE_PRESETS = {
    "coder": {
        "base_model": "deepseek-coder-v2:16b",
        "temperature": 0.3,
        "top_p": 0.9,
        "context_length": 8192,
        "system_prompt": """You are an expert software developer.
- Write clean, efficient, production-ready code
- Follow best practices and design patterns
- Include error handling and edge cases
- Comment only where logic isn't self-evident
- Prefer simple solutions over clever ones"""
    },
    "tester": {
        "base_model": "qwen3:4b",
        "temperature": 0.2,
        "top_p": 0.85,
        "context_length": 4096,
        "system_prompt": """You are a QA engineer and bug hunter.
- Write comprehensive test cases (unit, integration, E2E)
- Think about edge cases, race conditions, security
- Find bugs by analyzing code carefully
- Report bugs with: steps to reproduce, expected vs actual, severity
- Never assume code works — verify everything"""
    },
    "planner": {
        "base_model": "qwen3:8b",
        "temperature": 0.5,
        "top_p": 0.9,
        "context_length": 8192,
        "system_prompt": """You are a technical project manager and architect.
- Break complex tasks into clear, ordered steps
- Identify dependencies between tasks
- Estimate complexity (simple/medium/complex)
- Consider risks and fallback plans
- Output structured plans with milestones"""
    },
    "reviewer": {
        "base_model": "qwen3:14b",
        "temperature": 0.3,
        "top_p": 0.85,
        "context_length": 8192,
        "system_prompt": """You are a senior code reviewer.
- Review code for: bugs, security, performance, readability
- Rate severity: critical/high/medium/low/nitpick
- Suggest specific fixes, not vague advice
- Check for: OWASP top 10, race conditions, memory leaks
- Be thorough but constructive"""
    },
    "designer": {
        "base_model": "qwen3:8b",
        "temperature": 0.7,
        "top_p": 0.95,
        "context_length": 8192,
        "system_prompt": """You are a product designer and UX specialist.
- Design user flows and wireframes (describe in detail)
- Consider accessibility and mobile-first design
- Create design systems with consistent components
- Think about user psychology and conversion
- Output specifications developers can implement"""
    },
    "writer": {
        "base_model": "qwen3:4b",
        "temperature": 0.6,
        "top_p": 0.9,
        "context_length": 4096,
        "system_prompt": """You are a technical writer and content creator.
- Write clear documentation, README, guides
- Create marketing copy, blog posts, descriptions
- Adapt tone: technical, casual, professional, sales
- Structure content with headings, lists, examples
- SEO-aware writing when relevant"""
    },
    "researcher": {
        "base_model": "qwen3:14b",
        "temperature": 0.4,
        "top_p": 0.9,
        "context_length": 8192,
        "system_prompt": """You are a research analyst.
- Analyze problems from multiple angles
- Compare options with pros/cons tables
- Cite reasoning and evidence for conclusions
- Identify unknowns and suggest how to resolve them
- Summarize findings in actionable recommendations"""
    },
}


# ── Factory ─────────────────────────────────────────────────────────

class AgentFactory:
    """โรงงานผลิต AI Agent — บอกชื่อ+ความสามารถ ปั๊มออกมาให้"""

    def __init__(self):
        AGENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()

    def _load_registry(self):
        if REGISTRY_FILE.exists():
            try:
                return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {"agents": {}, "teams": {}}

    def _save_registry(self):
        REGISTRY_FILE.write_text(
            json.dumps(self.registry, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    # ── Create Agent ────────────────────────────────────────────────

    def create(self, name, expertise, preset=None, base_model=None,
               temperature=None, top_p=None, context_length=None,
               system_prompt=None):
        """สร้าง agent ใหม่

        Args:
            name: ชื่อ agent เช่น "game-coder", "api-tester"
            expertise: ความสามารถ เช่น "เขียน Unity C# script, game logic"
            preset: ใช้ preset สำเร็จรูป ("coder", "tester", "planner", ...)
            base_model: override base model
            temperature: override temperature
            top_p: override top_p
            context_length: override context length
            system_prompt: override system prompt (เพิ่มต่อจาก preset)
        """
        agent_name = f"ai-{name}"

        # Start from preset or defaults
        if preset and preset in ROLE_PRESETS:
            config = dict(ROLE_PRESETS[preset])
        elif preset:
            # Auto-detect best preset from name
            config = self._auto_detect_preset(name, expertise)
        else:
            config = self._auto_detect_preset(name, expertise)

        # Apply overrides
        if base_model:
            config["base_model"] = base_model
        if temperature is not None:
            config["temperature"] = temperature
        if top_p is not None:
            config["top_p"] = top_p
        if context_length is not None:
            config["context_length"] = context_length
        if system_prompt:
            config["system_prompt"] = config["system_prompt"] + "\n\n" + system_prompt

        # Generate Modelfile
        modelfile_content = MODELFILE_TEMPLATE.format(
            base_model=config["base_model"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            context_length=config["context_length"],
            role_name=name.replace("-", " ").title(),
            expertise=expertise,
            system_prompt=config["system_prompt"],
        )

        # Save Modelfile
        modelfile_path = AGENTS_DIR / f"{agent_name}.Modelfile"
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        # Check if base model exists
        if not self._model_exists(config["base_model"]):
            print(f"  Downloading base model: {config['base_model']}...")
            result = subprocess.run(
                ["ollama", "pull", config["base_model"]],
                capture_output=False
            )
            if result.returncode != 0:
                print(f"  ERROR: Failed to download {config['base_model']}")
                return False

        # Build with Ollama
        print(f"  Building {agent_name}...")
        result = subprocess.run(
            ["ollama", "create", agent_name, "-f", str(modelfile_path)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
            return False

        # Register
        self.registry["agents"][agent_name] = {
            "name": name,
            "expertise": expertise,
            "preset": preset or "auto",
            "base_model": config["base_model"],
            "temperature": config["temperature"],
            "context_length": config["context_length"],
            "created": datetime.now().isoformat(),
            "modelfile": str(modelfile_path),
        }
        self._save_registry()

        print(f"  Agent '{agent_name}' created successfully!")
        print(f"  Base: {config['base_model']} | Temp: {config['temperature']}")
        print(f"  Test: ollama run {agent_name}")
        return True

    def _auto_detect_preset(self, name, expertise):
        """เดา preset จากชื่อและความสามารถ"""
        text = f"{name} {expertise}".lower()

        if any(w in text for w in ["code", "dev", "program", "script", "เขียน"]):
            return dict(ROLE_PRESETS["coder"])
        if any(w in text for w in ["test", "bug", "qa", "ทดสอบ", "หา bug"]):
            return dict(ROLE_PRESETS["tester"])
        if any(w in text for w in ["plan", "manage", "วางแผน", "จัดการ"]):
            return dict(ROLE_PRESETS["planner"])
        if any(w in text for w in ["review", "audit", "ตรวจ"]):
            return dict(ROLE_PRESETS["reviewer"])
        if any(w in text for w in ["design", "ui", "ux", "ออกแบบ"]):
            return dict(ROLE_PRESETS["designer"])
        if any(w in text for w in ["write", "doc", "content", "blog", "เขียนบท"]):
            return dict(ROLE_PRESETS["writer"])
        if any(w in text for w in ["research", "analyze", "วิเคราะห์", "วิจัย"]):
            return dict(ROLE_PRESETS["researcher"])

        # Default: planner (most versatile)
        return dict(ROLE_PRESETS["planner"])

    def _model_exists(self, model_name):
        """เช็คว่า model มีใน ollama แล้วหรือยัง"""
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True
        )
        return model_name.split(":")[0] in result.stdout

    # ── Batch Create ────────────────────────────────────────────────

    def create_team(self, team_name, members):
        """สร้างทีม AI ทีเดียวหลายตัว

        Args:
            team_name: ชื่อทีม เช่น "game-studio"
            members: list of dict [{"name": "...", "expertise": "...", "preset": "..."}]

        Example:
            factory.create_team("game-studio", [
                {"name": "game-designer", "expertise": "ออกแบบ gameplay, level design"},
                {"name": "game-coder", "expertise": "เขียน game logic, Unity C#"},
                {"name": "game-artist", "expertise": "ออกแบบ art direction, UI"},
                {"name": "game-tester", "expertise": "ทดสอบเกม, หา bug, balance"},
            ])
        """
        print(f"\n{'='*60}")
        print(f"  CREATING TEAM: {team_name}")
        print(f"  Members: {len(members)}")
        print(f"{'='*60}\n")

        created = []
        failed = []

        for i, member in enumerate(members, 1):
            print(f"\n[{i}/{len(members)}] Creating: {member['name']}")
            print(f"  Expertise: {member['expertise']}")

            success = self.create(
                name=member["name"],
                expertise=member["expertise"],
                preset=member.get("preset"),
                base_model=member.get("base_model"),
                temperature=member.get("temperature"),
                system_prompt=member.get("system_prompt"),
            )

            if success:
                created.append(member["name"])
            else:
                failed.append(member["name"])

        # Register team
        self.registry["teams"][team_name] = {
            "members": [f"ai-{m['name']}" for m in members],
            "created": datetime.now().isoformat(),
        }
        self._save_registry()

        print(f"\n{'='*60}")
        print(f"  TEAM '{team_name}' COMPLETE")
        print(f"  Created: {len(created)} | Failed: {len(failed)}")
        if failed:
            print(f"  Failed: {', '.join(failed)}")
        print(f"{'='*60}\n")

        return created, failed

    # ── List / Delete / Info ────────────────────────────────────────

    def list_agents(self):
        """แสดง agents ทั้งหมดที่สร้างไว้"""
        if not self.registry["agents"]:
            print("No agents created yet.")
            return

        print(f"\n{'='*70}")
        print(f"  AI AGENTS ({len(self.registry['agents'])} total)")
        print(f"{'='*70}")
        print(f"  {'Name':<25} {'Base Model':<25} {'Preset':<10}")
        print(f"  {'-'*25} {'-'*25} {'-'*10}")

        for agent_name, info in self.registry["agents"].items():
            print(f"  {agent_name:<25} {info['base_model']:<25} {info['preset']:<10}")
            print(f"  {'':25} {info['expertise'][:45]}")
            print()

    def list_teams(self):
        """แสดงทีมทั้งหมด"""
        if not self.registry["teams"]:
            print("No teams created yet.")
            return

        print(f"\n{'='*60}")
        print(f"  AI TEAMS ({len(self.registry['teams'])} total)")
        print(f"{'='*60}")

        for team_name, info in self.registry["teams"].items():
            print(f"\n  Team: {team_name}")
            print(f"  Created: {info['created'][:10]}")
            print(f"  Members:")
            for member in info["members"]:
                agent_info = self.registry["agents"].get(member, {})
                expertise = agent_info.get("expertise", "unknown")
                print(f"    - {member}: {expertise[:50]}")

    def delete_agent(self, name):
        """ลบ agent"""
        agent_name = f"ai-{name}" if not name.startswith("ai-") else name

        # Remove from Ollama
        subprocess.run(["ollama", "rm", agent_name], capture_output=True)

        # Remove from registry
        if agent_name in self.registry["agents"]:
            del self.registry["agents"][agent_name]
            self._save_registry()

        # Remove Modelfile
        modelfile = AGENTS_DIR / f"{agent_name}.Modelfile"
        if modelfile.exists():
            modelfile.unlink()

        print(f"  Agent '{agent_name}' deleted.")

    def delete_team(self, team_name):
        """ลบทีมทั้งทีม"""
        if team_name not in self.registry["teams"]:
            print(f"  Team '{team_name}' not found.")
            return

        members = self.registry["teams"][team_name]["members"]
        for member in members:
            self.delete_agent(member)

        del self.registry["teams"][team_name]
        self._save_registry()
        print(f"  Team '{team_name}' and all members deleted.")

    # ── Presets Info ─────────────────────────────────────────────────

    def list_presets(self):
        """แสดง preset ที่ใช้ได้"""
        print(f"\n{'='*70}")
        print(f"  AVAILABLE PRESETS")
        print(f"{'='*70}")
        for name, config in ROLE_PRESETS.items():
            print(f"\n  {name}")
            print(f"    Base Model:  {config['base_model']}")
            print(f"    Temperature: {config['temperature']}")
            print(f"    Context:     {config['context_length']}")

    def list_base_models(self):
        """แสดง base model ที่แนะนำ"""
        print(f"\n{'='*70}")
        print(f"  RECOMMENDED BASE MODELS")
        print(f"{'='*70}")
        for model, info in BASE_MODELS.items():
            print(f"\n  {model} ({info['size']})")
            print(f"    Strength: {info['strength']}")
            print(f"    Best for: {', '.join(info['best_for'])}")


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Agent Factory — ปั๊ม AI Agent เฉพาะทาง",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # สร้าง agent ตัวเดียว
  python agent_factory.py create game-coder "เขียน game logic, Unity C#"

  # สร้าง agent พร้อมระบุ preset
  python agent_factory.py create api-tester "ทดสอบ REST API" --preset tester

  # สร้างทีมจากไฟล์ config
  python agent_factory.py team game-studio --config teams/game_studio.json

  # ดู agents ทั้งหมด
  python agent_factory.py list

  # ดู presets ที่ใช้ได้
  python agent_factory.py presets

  # ลบ agent
  python agent_factory.py delete game-coder
        """
    )

    sub = parser.add_subparsers(dest="command")

    # create
    p_create = sub.add_parser("create", help="สร้าง agent ใหม่")
    p_create.add_argument("name", help="ชื่อ agent")
    p_create.add_argument("expertise", help="ความสามารถ")
    p_create.add_argument("--preset", help="ใช้ preset (coder/tester/planner/...)")
    p_create.add_argument("--base-model", help="Override base model")
    p_create.add_argument("--temperature", type=float)
    p_create.add_argument("--context-length", type=int)

    # team
    p_team = sub.add_parser("team", help="สร้างทีมจาก config file")
    p_team.add_argument("name", help="ชื่อทีม")
    p_team.add_argument("--config", required=True, help="Path to team config JSON")

    # list
    sub.add_parser("list", help="แสดง agents ทั้งหมด")

    # teams
    sub.add_parser("teams", help="แสดงทีมทั้งหมด")

    # presets
    sub.add_parser("presets", help="แสดง presets ที่ใช้ได้")

    # models
    sub.add_parser("models", help="แสดง base models ที่แนะนำ")

    # delete
    p_del = sub.add_parser("delete", help="ลบ agent")
    p_del.add_argument("name", help="ชื่อ agent ที่จะลบ")

    # delete-team
    p_delt = sub.add_parser("delete-team", help="ลบทีมทั้งทีม")
    p_delt.add_argument("name", help="ชื่อทีมที่จะลบ")

    args = parser.parse_args()
    factory = AgentFactory()

    if args.command == "create":
        factory.create(
            name=args.name,
            expertise=args.expertise,
            preset=args.preset,
            base_model=args.base_model,
            temperature=args.temperature,
            context_length=args.context_length,
        )
    elif args.command == "team":
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Config file not found: {args.config}")
            sys.exit(1)
        team_config = json.loads(config_path.read_text(encoding="utf-8"))
        factory.create_team(args.name, team_config["members"])
    elif args.command == "list":
        factory.list_agents()
    elif args.command == "teams":
        factory.list_teams()
    elif args.command == "presets":
        factory.list_presets()
    elif args.command == "models":
        factory.list_base_models()
    elif args.command == "delete":
        factory.delete_agent(args.name)
    elif args.command == "delete-team":
        factory.delete_team(args.name)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
