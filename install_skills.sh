#!/bin/bash
# Установка скиллов в ~/.claude/skills/
# Использование: bash install_skills.sh

SKILLS_DIR="$HOME/.claude/skills"
SOURCE_DIR="$(cd "$(dirname "$0")/skills" && pwd)"

echo "Установка ML Pipeline скиллов..."
echo "Источник: $SOURCE_DIR"
echo "Назначение: $SKILLS_DIR"
echo ""

mkdir -p "$SKILLS_DIR"

SKILLS=("data-collector" "quality-guard" "auto-tagger" "smart-sampler" "ml-pipeline")

for skill in "${SKILLS[@]}"; do
    src="$SOURCE_DIR/$skill"
    dst="$SKILLS_DIR/$skill"

    if [ ! -d "$src" ]; then
        echo "  ✗ $skill — папка не найдена: $src"
        continue
    fi

    rm -rf "$dst"
    cp -r "$src" "$dst"
    echo "  ✓ $skill → $dst"
done

echo ""
echo "Установлено скиллов: ${#SKILLS[@]}"
echo ""
echo "Доступные команды в Claude Code:"
echo "  /data-collector  \"<тема>\""
echo "  /quality-guard"
echo "  /auto-tagger     --classes \"<классы>\" --task \"<описание>\""
echo "  /smart-sampler"
echo "  /ml-pipeline     \"<тема>\" --classes \"<классы>\" --task \"<описание>\""
