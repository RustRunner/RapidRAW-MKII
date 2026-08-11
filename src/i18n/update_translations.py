import json
from pathlib import Path

LOCALES_DIR = Path("./locales")

TRANSLATIONS = {
    "de": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "Keine Bewegungsunschärfe-Richtung erkannt - Länge und Winkel manuell einstellen.",
                    "estimateFailedDefocus": "Kein Defokus-Radius erkannt - Radius manuell einstellen.",
                    "estimateFailedGaussian": "Gaußsche Unschärfe nicht messbar - Sigma manuell einstellen.",
                }
            }
        }
    },
    "en": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "Could not detect a motion-blur direction - set length and angle manually.",
                    "estimateFailedDefocus": "Could not detect a defocus radius - set the radius manually.",
                    "estimateFailedGaussian": "Could not measure a gaussian blur level - set the sigma manually.",
                }
            }
        }
    },
    "es": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "No se detectó la dirección del desenfoque de movimiento: ajusta longitud y ángulo manualmente.",
                    "estimateFailedDefocus": "No se detectó el radio de desenfoque: ajusta el radio manualmente.",
                    "estimateFailedGaussian": "No se pudo medir el nivel de desenfoque gaussiano: ajusta el sigma manualmente.",
                }
            }
        }
    },
    "fr": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "Direction du flou de bougé non détectée - réglez la longueur et l'angle manuellement.",
                    "estimateFailedDefocus": "Rayon de défocalisation non détecté - réglez le rayon manuellement.",
                    "estimateFailedGaussian": "Niveau de flou gaussien non mesurable - réglez le sigma manuellement.",
                }
            }
        }
    },
    "it": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "Direzione del mosso non rilevata: imposta lunghezza e angolo manualmente.",
                    "estimateFailedDefocus": "Raggio di sfocatura non rilevato: imposta il raggio manualmente.",
                    "estimateFailedGaussian": "Livello di sfocatura gaussiana non misurabile: imposta il sigma manualmente.",
                }
            }
        }
    },
    "ja": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "ブレの方向を検出できませんでした。長さと角度を手動で設定してください。",
                    "estimateFailedDefocus": "デフォーカス半径を検出できませんでした。半径を手動で設定してください。",
                    "estimateFailedGaussian": "ガウスぼかしの強さを測定できませんでした。シグマを手動で設定してください。",
                }
            }
        }
    },
    "ko": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "모션 블러 방향을 감지하지 못했습니다. 길이와 각도를 직접 설정하세요.",
                    "estimateFailedDefocus": "디포커스 반경을 감지하지 못했습니다. 반경을 직접 설정하세요.",
                    "estimateFailedGaussian": "가우시안 블러 정도를 측정하지 못했습니다. 시그마를 직접 설정하세요.",
                }
            }
        }
    },
    "pl": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "Nie wykryto kierunku rozmycia ruchu - ustaw długość i kąt ręcznie.",
                    "estimateFailedDefocus": "Nie wykryto promienia rozogniskowania - ustaw promień ręcznie.",
                    "estimateFailedGaussian": "Nie udało się zmierzyć poziomu rozmycia gaussowskiego - ustaw sigmę ręcznie.",
                }
            }
        }
    },
    "pt": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "Direção do borrão de movimento não detectada - ajuste comprimento e ângulo manualmente.",
                    "estimateFailedDefocus": "Raio de desfoque não detectado - ajuste o raio manualmente.",
                    "estimateFailedGaussian": "Não foi possível medir o nível de desfoque gaussiano - ajuste o sigma manualmente.",
                }
            }
        }
    },
    "ru": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "Направление смаза не обнаружено - задайте длину и угол вручную.",
                    "estimateFailedDefocus": "Радиус расфокусировки не обнаружен - задайте радиус вручную.",
                    "estimateFailedGaussian": "Не удалось измерить уровень гауссова размытия - задайте сигму вручную.",
                }
            }
        }
    },
    "zh-CN": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "未能检测到运动模糊方向，请手动设置长度和角度。",
                    "estimateFailedDefocus": "未能检测到散焦半径，请手动设置半径。",
                    "estimateFailedGaussian": "未能测量高斯模糊程度，请手动设置 Sigma。",
                }
            }
        }
    },
    "zh-TW": {
        "editor": {
            "adjustments": {
                "blurRecovery": {
                    "estimateFailedMotion": "未能偵測到運動模糊方向，請手動設定長度與角度。",
                    "estimateFailedDefocus": "未能偵測到散焦半徑，請手動設定半徑。",
                    "estimateFailedGaussian": "未能測量高斯模糊程度，請手動設定 Sigma。",
                }
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
