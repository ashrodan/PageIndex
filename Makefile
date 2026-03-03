SHELL := /bin/bash

PYTHON ?= .venv/bin/python
SKILL_DIR ?= .claude/skills/pageindex-search
SKILL_NAME ?= $(notdir $(SKILL_DIR))
DIST_DIR ?= .dist/skills
BUNDLE ?= $(DIST_DIR)/$(SKILL_NAME).zip
SET_DEFAULT ?= true

.PHONY: help skill-zip skill-upload-new skill-upload-version skill-list

help:
	@echo "Targets:"
	@echo "  make skill-zip"
	@echo "    Zip SKILL_DIR into BUNDLE (includes one top-level skill folder)."
	@echo ""
	@echo "  make skill-upload-new OPENAI_API_KEY=... [SKILL_DIR=...]"
	@echo "    Create a new hosted OpenAI skill from the zipped bundle."
	@echo ""
	@echo "  make skill-upload-version OPENAI_API_KEY=... SKILL_ID=skill_... [SET_DEFAULT=true|false]"
	@echo "    Upload a new immutable version for an existing skill."
	@echo ""
	@echo "  make skill-list OPENAI_API_KEY=..."
	@echo "    List current project skills."

skill-zip:
	@test -d "$(SKILL_DIR)" || (echo "Skill directory not found: $(SKILL_DIR)" && exit 1)
	@mkdir -p "$(DIST_DIR)"
	@cd "$(dir $(SKILL_DIR))" && zip -r "$(abspath $(BUNDLE))" "$(notdir $(SKILL_DIR))" >/dev/null
	@echo "Created bundle: $(BUNDLE)"

skill-upload-new: skill-zip
	@test -n "$$OPENAI_API_KEY" || (echo "Set OPENAI_API_KEY" && exit 1)
	@BUNDLE="$(abspath $(BUNDLE))" "$(PYTHON)" -c "import json, os; from openai import OpenAI; client = OpenAI(api_key=os.environ['OPENAI_API_KEY']); bundle = os.environ['BUNDLE']; f = open(bundle, 'rb'); skill = client.skills.create(files=f); f.close(); print(json.dumps({'skill_id': skill.id, 'name': getattr(skill, 'name', None), 'default_version': getattr(skill, 'default_version', None), 'latest_version': getattr(skill, 'latest_version', None)}, indent=2))"

skill-upload-version: skill-zip
	@test -n "$$OPENAI_API_KEY" || (echo "Set OPENAI_API_KEY" && exit 1)
	@test -n "$(SKILL_ID)" || (echo "Set SKILL_ID=skill_..." && exit 1)
	@BUNDLE="$(abspath $(BUNDLE))" SKILL_ID="$(SKILL_ID)" SET_DEFAULT="$(SET_DEFAULT)" "$(PYTHON)" -c "import json, os; from openai import OpenAI; client = OpenAI(api_key=os.environ['OPENAI_API_KEY']); bundle = os.environ['BUNDLE']; skill_id = os.environ['SKILL_ID']; set_default = os.environ.get('SET_DEFAULT', 'true').lower() in ('1', 'true', 'yes'); f = open(bundle, 'rb'); version = client.skills.versions.create(skill_id=skill_id, files=f, default=set_default); f.close(); print(json.dumps({'skill_id': skill_id, 'version_id': version.id, 'version': getattr(version, 'version', None), 'default': getattr(version, 'default', None)}, indent=2))"

skill-list:
	@test -n "$$OPENAI_API_KEY" || (echo "Set OPENAI_API_KEY" && exit 1)
	@"$(PYTHON)" -c "import json, os; from openai import OpenAI; client = OpenAI(api_key=os.environ['OPENAI_API_KEY']); skills = client.skills.list(limit=100); rows = [{'id': s.id, 'name': getattr(s, 'name', None), 'default_version': getattr(s, 'default_version', None), 'latest_version': getattr(s, 'latest_version', None)} for s in skills.data]; print(json.dumps(rows, indent=2))"
