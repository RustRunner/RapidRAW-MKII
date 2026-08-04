import json
from pathlib import Path

LOCALES_DIR = Path("./locales")

TRANSLATIONS = {
    "de": {
        "export": {
            "sections": {"callout": "Infobox"},
            "callout": {
                "addCallout": "Infobox hinzufügen",
                "notesPlaceholder": "Notizen, die in den Export gerendert werden…",
                "prefillMetadata": "Aus Metadaten übernehmen",
                "insertTemplate": "Vorlage einfügen",
                "saveTemplate": "Als Standard speichern",
                "templateSaved": "Gespeichert",
                "mgrsCoords": "MGRS-Koordinaten",
                "textSize": "Textgröße",
                "spacing": "Abstand",
                "opacity": "Deckkraft",
                "opacityHint": "Bei 0 % Deckkraft wird nur der Text gezeichnet — ohne Box.",
                "previewText": "Vorschau",
            },
        }
    },
    "en": {
        "export": {
            "sections": {"callout": "Callout Box"},
            "callout": {
                "addCallout": "Add Callout Box",
                "notesPlaceholder": "Notes rendered onto the export…",
                "prefillMetadata": "Prefill from Metadata",
                "insertTemplate": "Insert Template",
                "saveTemplate": "Save as Default",
                "templateSaved": "Saved",
                "mgrsCoords": "MGRS Coordinates",
                "textSize": "Text Size",
                "spacing": "Spacing",
                "opacity": "Opacity",
                "opacityHint": "At 0% opacity, only the text is drawn — no box.",
                "previewText": "Preview",
            },
        }
    },
    "es": {
        "export": {
            "sections": {"callout": "Cuadro de texto"},
            "callout": {
                "addCallout": "Añadir cuadro de texto",
                "notesPlaceholder": "Notas que se renderizan en la exportación…",
                "prefillMetadata": "Rellenar desde metadatos",
                "insertTemplate": "Insertar plantilla",
                "saveTemplate": "Guardar como predeterminada",
                "templateSaved": "Guardado",
                "mgrsCoords": "Coordenadas MGRS",
                "textSize": "Tamaño del texto",
                "spacing": "Espaciado",
                "opacity": "Opacidad",
                "opacityHint": "Con opacidad al 0 % solo se dibuja el texto, sin cuadro.",
                "previewText": "Vista previa",
            },
        }
    },
    "fr": {
        "export": {
            "sections": {"callout": "Encadré"},
            "callout": {
                "addCallout": "Ajouter un encadré",
                "notesPlaceholder": "Notes rendues sur l'export…",
                "prefillMetadata": "Préremplir depuis les métadonnées",
                "insertTemplate": "Insérer le modèle",
                "saveTemplate": "Enregistrer par défaut",
                "templateSaved": "Enregistré",
                "mgrsCoords": "Coordonnées MGRS",
                "textSize": "Taille du texte",
                "spacing": "Espacement",
                "opacity": "Opacité",
                "opacityHint": "À 0 % d'opacité, seul le texte est dessiné, sans encadré.",
                "previewText": "Aperçu",
            },
        }
    },
    "it": {
        "export": {
            "sections": {"callout": "Riquadro di testo"},
            "callout": {
                "addCallout": "Aggiungi riquadro di testo",
                "notesPlaceholder": "Note renderizzate nell'esportazione…",
                "prefillMetadata": "Precompila dai metadati",
                "insertTemplate": "Inserisci modello",
                "saveTemplate": "Salva come predefinito",
                "templateSaved": "Salvato",
                "mgrsCoords": "Coordinate MGRS",
                "textSize": "Dimensione testo",
                "spacing": "Spaziatura",
                "opacity": "Opacità",
                "opacityHint": "Con opacità allo 0 % viene disegnato solo il testo, senza riquadro.",
                "previewText": "Anteprima",
            },
        }
    },
    "ja": {
        "export": {
            "sections": {"callout": "注釈ボックス"},
            "callout": {
                "addCallout": "注釈ボックスを追加",
                "notesPlaceholder": "書き出し画像に描画されるメモ…",
                "prefillMetadata": "メタデータから入力",
                "insertTemplate": "テンプレートを挿入",
                "saveTemplate": "デフォルトとして保存",
                "templateSaved": "保存しました",
                "mgrsCoords": "MGRS座標",
                "textSize": "文字サイズ",
                "spacing": "間隔",
                "opacity": "不透明度",
                "opacityHint": "不透明度 0% ではボックスなしで文字のみ描画されます。",
                "previewText": "プレビュー",
            },
        }
    },
    "ko": {
        "export": {
            "sections": {"callout": "설명 상자"},
            "callout": {
                "addCallout": "설명 상자 추가",
                "notesPlaceholder": "내보내기 이미지에 렌더링될 메모…",
                "prefillMetadata": "메타데이터에서 채우기",
                "insertTemplate": "템플릿 삽입",
                "saveTemplate": "기본값으로 저장",
                "templateSaved": "저장됨",
                "mgrsCoords": "MGRS 좌표",
                "textSize": "텍스트 크기",
                "spacing": "간격",
                "opacity": "불투명도",
                "opacityHint": "불투명도 0%에서는 상자 없이 텍스트만 그려집니다.",
                "previewText": "미리보기",
            },
        }
    },
    "pl": {
        "export": {
            "sections": {"callout": "Ramka tekstowa"},
            "callout": {
                "addCallout": "Dodaj ramkę tekstową",
                "notesPlaceholder": "Notatki renderowane na eksporcie…",
                "prefillMetadata": "Wypełnij z metadanych",
                "insertTemplate": "Wstaw szablon",
                "saveTemplate": "Zapisz jako domyślny",
                "templateSaved": "Zapisano",
                "mgrsCoords": "Współrzędne MGRS",
                "textSize": "Rozmiar tekstu",
                "spacing": "Odstępy",
                "opacity": "Krycie",
                "opacityHint": "Przy kryciu 0 % rysowany jest tylko tekst, bez ramki.",
                "previewText": "Podgląd",
            },
        }
    },
    "pt": {
        "export": {
            "sections": {"callout": "Caixa de texto"},
            "callout": {
                "addCallout": "Adicionar caixa de texto",
                "notesPlaceholder": "Notas renderizadas na exportação…",
                "prefillMetadata": "Preencher com metadados",
                "insertTemplate": "Inserir modelo",
                "saveTemplate": "Salvar como padrão",
                "templateSaved": "Salvo",
                "mgrsCoords": "Coordenadas MGRS",
                "textSize": "Tamanho do texto",
                "spacing": "Espaçamento",
                "opacity": "Opacidade",
                "opacityHint": "Com opacidade em 0 %, apenas o texto é desenhado, sem caixa.",
                "previewText": "Visualização",
            },
        }
    },
    "ru": {
        "export": {
            "sections": {"callout": "Текстовый блок"},
            "callout": {
                "addCallout": "Добавить текстовый блок",
                "notesPlaceholder": "Заметки, отображаемые на экспортируемом изображении…",
                "prefillMetadata": "Заполнить из метаданных",
                "insertTemplate": "Вставить шаблон",
                "saveTemplate": "Сохранить как шаблон по умолчанию",
                "templateSaved": "Сохранено",
                "mgrsCoords": "Координаты MGRS",
                "textSize": "Размер текста",
                "spacing": "Отступы",
                "opacity": "Непрозрачность",
                "opacityHint": "При непрозрачности 0 % отображается только текст, без подложки.",
                "previewText": "Предпросмотр",
            },
        }
    },
    "zh-CN": {
        "export": {
            "sections": {"callout": "标注框"},
            "callout": {
                "addCallout": "添加标注框",
                "notesPlaceholder": "将渲染到导出图像上的备注…",
                "prefillMetadata": "从元数据填充",
                "insertTemplate": "插入模板",
                "saveTemplate": "保存为默认",
                "templateSaved": "已保存",
                "mgrsCoords": "MGRS 坐标",
                "textSize": "文字大小",
                "spacing": "间距",
                "opacity": "不透明度",
                "opacityHint": "不透明度为 0% 时仅绘制文字，不显示底框。",
                "previewText": "预览",
            },
        }
    },
    "zh-TW": {
        "export": {
            "sections": {"callout": "標註框"},
            "callout": {
                "addCallout": "新增標註框",
                "notesPlaceholder": "將算繪到匯出影像上的備註…",
                "prefillMetadata": "從中繼資料填入",
                "insertTemplate": "插入範本",
                "saveTemplate": "儲存為預設",
                "templateSaved": "已儲存",
                "mgrsCoords": "MGRS 座標",
                "textSize": "文字大小",
                "spacing": "間距",
                "opacity": "不透明度",
                "opacityHint": "不透明度為 0% 時僅繪製文字，不顯示底框。",
                "previewText": "預覽",
            },
        }
    },
}

def deep_merge(target: dict, source: dict):
    """Recursively merges source dict into target dict."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = target.setdefault(key, {})
            if isinstance(node, dict):
                deep_merge(node, value)
        else:
            target[key] = value

def sort_dict_recursively(item):
    if isinstance(item, dict):
        return {k: sort_dict_recursively(v) for k, v in sorted(item.items())}
    elif isinstance(item, list):
        return [sort_dict_recursively(x) for x in item]
    return item

def update_json_file(file_path: Path, trans: dict):
    if not file_path.exists():
        print(f"Skipping: {file_path.name} (File not found)")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error parsing JSON in {file_path.name}. Skipping.")
        return

    deep_merge(data, trans)
    sorted_data = sort_dict_recursively(data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Updated and Sorted: {file_path.name}")

def main():
    print("Starting translation updates...")
    for locale, trans in TRANSLATIONS.items():
        update_json_file(LOCALES_DIR / f"{locale}.json", trans)
    print("Done.")

if __name__ == "__main__":
    main()
