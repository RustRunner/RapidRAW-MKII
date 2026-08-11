import json
from pathlib import Path

LOCALES_DIR = Path("./locales")

TRANSLATIONS = {
    "de": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Aktivieren"},
                "glareRecovery": {"enable": "Aktivieren"},
                "lowlight": {
                    "sensitivity": "Empfindlichkeit",
                    "description": "Werkzeuge für High-ISO- und Langzeitaufnahmen. Hot-Pixel-Entfernung und Entrauschen als Live-Vorschau.",
                },
            }
        }
    },
    "en": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Enable"},
                "glareRecovery": {"enable": "Enable"},
                "lowlight": {
                    "sensitivity": "Sensitivity",
                    "description": "Recovery tools for high-ISO and long-exposure shots. Hot-pixel removal and denoising preview live.",
                },
            }
        }
    },
    "es": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Activar"},
                "glareRecovery": {"enable": "Activar"},
                "lowlight": {
                    "sensitivity": "Sensibilidad",
                    "description": "Herramientas de recuperación para tomas con ISO alto y larga exposición. La eliminación de píxeles calientes y la reducción de ruido se previsualizan en vivo.",
                },
            }
        }
    },
    "fr": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Activer"},
                "glareRecovery": {"enable": "Activer"},
                "lowlight": {
                    "sensitivity": "Sensibilité",
                    "description": "Outils de récupération pour les prises à haut ISO et longue exposition. La correction des pixels chauds et le débruitage s'affichent en direct.",
                },
            }
        }
    },
    "it": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Attiva"},
                "glareRecovery": {"enable": "Attiva"},
                "lowlight": {
                    "sensitivity": "Sensibilità",
                    "description": "Strumenti di recupero per scatti ad alto ISO e lunga esposizione. Rimozione hot pixel e riduzione del rumore in anteprima dal vivo.",
                },
            }
        }
    },
    "ja": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "有効化"},
                "glareRecovery": {"enable": "有効化"},
                "lowlight": {
                    "sensitivity": "感度",
                    "description": "高ISO・長時間露光向けの回復ツール。ホットピクセル除去とノイズ除去はライブプレビュー。",
                },
            }
        }
    },
    "ko": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "활성화"},
                "glareRecovery": {"enable": "활성화"},
                "lowlight": {
                    "sensitivity": "감도",
                    "description": "고감도·장노출 촬영을 위한 복구 도구입니다. 핫픽셀 제거와 노이즈 제거는 실시간 미리보기됩니다.",
                },
            }
        }
    },
    "pl": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Włącz"},
                "glareRecovery": {"enable": "Włącz"},
                "lowlight": {
                    "sensitivity": "Czułość",
                    "description": "Narzędzia ratunkowe dla zdjęć z wysokim ISO i długim czasem naświetlania. Usuwanie gorących pikseli i odszumianie działają na żywo.",
                },
            }
        }
    },
    "pt": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Ativar"},
                "glareRecovery": {"enable": "Ativar"},
                "lowlight": {
                    "sensitivity": "Sensibilidade",
                    "description": "Ferramentas de recuperação para fotos com ISO alto e longa exposição. A remoção de hot pixels e a redução de ruído têm pré-visualização ao vivo.",
                },
            }
        }
    },
    "ru": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Включить"},
                "glareRecovery": {"enable": "Включить"},
                "lowlight": {
                    "sensitivity": "Чувствительность",
                    "description": "Инструменты восстановления для снимков с высоким ISO и длинной выдержкой. Удаление горячих пикселей и шумоподавление работают в реальном времени.",
                },
            }
        }
    },
    "zh-CN": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "启用"},
                "glareRecovery": {"enable": "启用"},
                "lowlight": {
                    "sensitivity": "灵敏度",
                    "description": "针对高 ISO 和长曝光照片的修复工具。热像素移除和降噪支持实时预览。",
                },
            }
        }
    },
    "zh-TW": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "啟用"},
                "glareRecovery": {"enable": "啟用"},
                "lowlight": {
                    "sensitivity": "靈敏度",
                    "description": "針對高 ISO 與長曝光照片的修復工具。熱像素移除與降噪支援即時預覽。",
                },
            }
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
