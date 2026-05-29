# scripts/ — 舊腳本(逐步收薄,邏輯已遷往 src/)

2026-05-30 大重構後,核心邏輯改放 `src/`,統一入口 `cli.py`(規則見 `docs/reference/整理規則.md`)。
本資料夾的腳本**保留作歷史/參考**,但**事實來源已是 `src/`**。新工作請用 `cli.py`,別再改這裡。

## 已遷移(src/ 為準,scripts/ 僅留存)
| 舊 script | 新位置 | cli 入口 |
| --- | --- | --- |
| `render_stage1.py` | `src/render/classify.py` | (Blender 內跑) |
| `render_pan_head.py` | `src/render/patches.py` | `cli.py render-patches` |
| `gen_backgrounds.py` | `src/render/backgrounds.py` | `cli.py gen-backgrounds` |
| `composite.py` | `src/data/generator.py`(改 runtime 確定性生成) | `cli.py freeze-eval` / `samples` |
| `train_stage4.py` | `src/{models,train,eval}/` | `cli.py train` |
| `eval_stage4.py` | `src/eval/metrics.py` | `cli.py eval` |

> 注:`composite.py` → `generator.py` 不是 1:1 —— 舊版單一 RNG 串流預存 1000 張;
> 新版 `scene(i)=f(i,config)` per-i 確定性、當場生成、不存訓練圖。

## 尚未遷移(scripts/ 獨有,需要時再處理)
- **報告產生**:通用 run 報告已接 `cli.py report`(`src/eval/report.py`)。
  `build_stage3_html.py` / `build_stage4_html.py` / `rework_figs_ef.py` 是**S4 專屬敘事報告**
  (手工圖 + 寫死數字 + prose),性質不同,保留為一次性。
- **舊 stage 訓練/評估**:`train_stage2_runner.py` / `train_stage3.py` / `eval_stage2.py` / `eval_stage3.py` / `eval_stage4_compare.py`
- **素材/工具**:`download_ambientcg.py` / `preview_parts.py` / `preview_scenes.py` / `preview_bg_candidates.py` / `gen_black_bg_test.py`
- **sanity 檢查**:`check_bend_axis.py` / `check_remesh_strength.py`(Blender 側一次性驗證)
