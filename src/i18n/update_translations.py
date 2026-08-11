import json
from pathlib import Path

LOCALES_DIR = Path("./locales")

TRANSLATIONS = {
    "de": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Aktivieren", "shared": "Gemeinsam"},
                "glareRecovery": {"enable": "Aktivieren"},
                "lowlight": {
                    "denoise": "Adaptives Entrauschen",
                    "denoiseDescription": "Messbasiertes, kantenschonendes Entrauschen: Schätzen misst das Rauschen dieses Bildes und setzt die Stärken. Ergebnis bei 100 % Zoom beurteilen - die verkleinerte Vorschau verbirgt feines Korn.",
                    "sensitivity": "Empfindlichkeit",
                    "description": "Werkzeuge für High-ISO- und Langzeitaufnahmen. Hot-Pixel-Entfernung und Entrauschen als Live-Vorschau.",
                },
            }
        }
    },
    "en": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Enable", "shared": "Shared"},
                "glareRecovery": {"enable": "Enable"},
                "lowlight": {
                    "denoise": "Adaptive Denoise",
                    "denoiseDescription": "Edge-aware denoising tuned to this image's measured noise: Estimate reads the noise floor and sets the strengths. Judge results at 100% zoom - a fit-to-screen preview hides fine grain.",
                    "sensitivity": "Sensitivity",
                    "description": "Recovery tools for high-ISO and long-exposure shots. Hot-pixel removal and denoising preview live.",
                },
            }
        }
    },
    "es": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Activar", "shared": "Compartidos"},
                "glareRecovery": {"enable": "Activar"},
                "lowlight": {
                    "denoise": "Reducción de ruido adaptativa",
                    "denoiseDescription": "Reducción de ruido medida y que preserva bordes: Estimar mide el ruido de esta imagen y ajusta las intensidades. Evalúa el resultado al 100 % de zoom: la vista ajustada oculta el grano fino.",
                    "sensitivity": "Sensibilidad",
                    "description": "Herramientas de recuperación para tomas con ISO alto y larga exposición. La eliminación de píxeles calientes y la reducción de ruido se previsualizan en vivo.",
                },
            }
        }
    },
    "fr": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Activer", "shared": "Partagés"},
                "glareRecovery": {"enable": "Activer"},
                "lowlight": {
                    "denoise": "Débruitage adaptatif",
                    "denoiseDescription": "Débruitage mesuré préservant les contours : Estimer mesure le bruit de cette image et règle les intensités. Jugez le résultat au zoom 100 % - l'aperçu ajusté masque le grain fin.",
                    "sensitivity": "Sensibilité",
                    "description": "Outils de récupération pour les prises à haut ISO et longue exposition. La correction des pixels chauds et le débruitage s'affichent en direct.",
                },
            }
        }
    },
    "it": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Attiva", "shared": "Condivisi"},
                "glareRecovery": {"enable": "Attiva"},
                "lowlight": {
                    "denoise": "Riduzione rumore adattiva",
                    "denoiseDescription": "Riduzione del rumore misurata e rispettosa dei bordi: Stima misura il rumore di questa immagine e imposta le intensità. Valuta il risultato allo zoom 100% - l'anteprima adattata nasconde la grana fine.",
                    "sensitivity": "Sensibilità",
                    "description": "Strumenti di recupero per scatti ad alto ISO e lunga esposizione. Rimozione hot pixel e riduzione del rumore in anteprima dal vivo.",
                },
            }
        }
    },
    "ja": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "有効化", "shared": "共通"},
                "glareRecovery": {"enable": "有効化"},
                "lowlight": {
                    "denoise": "適応ノイズ除去",
                    "denoiseDescription": "実測ベースのエッジ保護ノイズ除去。推定がこの画像のノイズを測定して強さを設定します。結果は100%ズームで確認してください。縮小表示では細かいノイズが見えません。",
                    "sensitivity": "感度",
                    "description": "高ISO・長時間露光向けの回復ツール。ホットピクセル除去とノイズ除去はライブプレビュー。",
                },
            }
        }
    },
    "ko": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "활성화", "shared": "공통"},
                "glareRecovery": {"enable": "활성화"},
                "lowlight": {
                    "denoise": "적응형 노이즈 제거",
                    "denoiseDescription": "측정 기반 엣지 보호 노이즈 제거입니다. 추정이 이 이미지의 노이즈를 측정해 강도를 설정합니다. 결과는 100% 확대에서 확인하세요. 축소된 미리보기에서는 미세한 노이즈가 보이지 않습니다.",
                    "sensitivity": "감도",
                    "description": "고감도·장노출 촬영을 위한 복구 도구입니다. 핫픽셀 제거와 노이즈 제거는 실시간 미리보기됩니다.",
                },
            }
        }
    },
    "pl": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Włącz", "shared": "Wspólne"},
                "glareRecovery": {"enable": "Włącz"},
                "lowlight": {
                    "denoise": "Adaptacyjne odszumianie",
                    "denoiseDescription": "Pomiarowe, chroniące krawędzie odszumianie: Szacuj mierzy szum tego zdjęcia i ustawia siły. Oceniaj wynik przy powiększeniu 100% - dopasowany podgląd ukrywa drobne ziarno.",
                    "sensitivity": "Czułość",
                    "description": "Narzędzia ratunkowe dla zdjęć z wysokim ISO i długim czasem naświetlania. Usuwanie gorących pikseli i odszumianie działają na żywo.",
                },
            }
        }
    },
    "pt": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Ativar", "shared": "Compartilhados"},
                "glareRecovery": {"enable": "Ativar"},
                "lowlight": {
                    "denoise": "Redução de ruído adaptativa",
                    "denoiseDescription": "Redução de ruído medida e que preserva bordas: Estimar mede o ruído desta imagem e define as intensidades. Avalie o resultado com zoom de 100% - a pré-visualização ajustada oculta o grão fino.",
                    "sensitivity": "Sensibilidade",
                    "description": "Ferramentas de recuperação para fotos com ISO alto e longa exposição. A remoção de hot pixels e a redução de ruído têm pré-visualização ao vivo.",
                },
            }
        }
    },
    "ru": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "Включить", "shared": "Общие"},
                "glareRecovery": {"enable": "Включить"},
                "lowlight": {
                    "denoise": "Адаптивное шумоподавление",
                    "denoiseDescription": "Измеренное шумоподавление с защитой контуров: Оценка измеряет шум этого снимка и задаёт силу. Оценивайте результат при 100% масштабе - уменьшенный просмотр скрывает мелкое зерно.",
                    "sensitivity": "Чувствительность",
                    "description": "Инструменты восстановления для снимков с высоким ISO и длинной выдержкой. Удаление горячих пикселей и шумоподавление работают в реальном времени.",
                },
            }
        }
    },
    "zh-CN": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "启用", "shared": "共用"},
                "glareRecovery": {"enable": "启用"},
                "lowlight": {
                    "denoise": "自适应降噪",
                    "denoiseDescription": "实测边缘保护降噪：估算会测量此图像的噪点并设置强度。请在 100% 缩放下查看效果——适应窗口的预览会掩盖细微噪点。",
                    "sensitivity": "灵敏度",
                    "description": "针对高 ISO 和长曝光照片的修复工具。热像素移除和降噪支持实时预览。",
                },
            }
        }
    },
    "zh-TW": {
        "editor": {
            "adjustments": {
                "blurRecovery": {"enable": "啟用", "shared": "共用"},
                "glareRecovery": {"enable": "啟用"},
                "lowlight": {
                    "denoise": "自適應降噪",
                    "denoiseDescription": "實測邊緣保護降噪：估算會測量此影像的噪點並設定強度。請在 100% 縮放下檢視效果——符合視窗的預覽會掩蓋細微噪點。",
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
