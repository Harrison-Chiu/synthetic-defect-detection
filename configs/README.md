# configs/

生成 / 訓練參數的存放處,**git 追蹤**(整理規則 §3:版控核心 = 零件 patch + 生成 config + 凍結 eval set)。

## 為什麼追蹤
場景是純函數 `scene(i) = f(i, config)`,所以「config 數值」就是資料集的身分證:
固定 config + 固定 patch 池 → 可完全重生同一批場景,不必把 1000 張訓練圖上 git。

## 現況
參數目前以 dataclass 預設值為單一事實來源:
- 生成:`src/data/config.py` 的 `GenConfig`(`DEFAULT_CONFIG`)
- 訓練:`src/train/config.py` 的 `TrainConfig`(`DEFAULT_TRAIN_CONFIG`)

兩者預設**忠實對齊 Stage 4**。數值調校(displace / HDRI pool / step 數等)待 S4 review 後再定。

## 之後
待 `cli.py --config <path>` 載入功能完成,這裡放序列化的 `*.json`(每次實驗一份),
檔名即實驗指紋,與 `output/runs/<stage_tag>/` 對應。
