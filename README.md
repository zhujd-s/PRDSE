# RLDSE

一、投稿TCAD

1、会议转投的要求：

- 30%以上的实质性扩展
- 文本重复率低于40%



2、期刊的要求

- IEEE期刊格式，双栏14页

原先是双栏8页，在原来的基础上增加即可

- 

二、跨电脑继续开发说明

1、建议先同步到 GitHub

- 需要同步的核心文件：
  - `test12_RL_RLDSE_MAESTRO.py`
  - `util/metric_controller.py`
  - `util/context_buffer.py`
  - `util/evaluation_maestro.py`
  - `util/config_analyzer.py`
  - `tool08_ExtendedBaseline.py`
  - `data/baseline_extended_*.csv`
  - `.gitignore`

- 不建议同步的实验输出：
  - `record/`
  - `__pycache__/`

2、当前 reward scheduler 相关状态

- `test12_RL_RLDSE_MAESTRO.py` 已接入 Transformer controller
- 当前 `lambda_ext` 使用：
  - `schedule_lambda_ext + learned delta`
  - 其中 `delta` 为对称 `+-0.1` 微调
- `controller_advantage` 由 `objectvalue` 改善决定
- `controller_loss` 当前已回退为更平滑的 reward-based 形式：
  - `controller_loss = -controller_advantage * final_reward + reg`

3、新增 intrinsic metrics

- 新增了 3 个指标：
  - `throughput`
  - `throughput_per_energy`
  - `offchip_bw_req`
- 目前只接入了 intrinsic reward，没有加入 controller context
- 这 3 个指标现在使用固定 baseline 归一化，不再使用 running max

4、扩展 baseline 文件

- `tool08_ExtendedBaseline.py` 会基于 `data/initial_data_warehouse_*.csv`
- 对六个模型分别固定采样并重跑 MAESTRO
- 在 `data/` 下生成：
  - `baseline_extended_VGG16.csv`
  - `baseline_extended_MobileNetV2.csv`
  - `baseline_extended_MnasNet.csv`
  - `baseline_extended_ResNet50.csv`
  - `baseline_extended_Transformer.csv`
  - `baseline_extended_GNMT.csv`
- `config_analyzer.py` 已支持自动读取这些扩展 baseline

5、另一台电脑上的推荐顺序

1. `git clone` 或 `git pull`
2. 准备好 MAESTRO 可执行环境
3. 如需重建扩展 baseline，运行：
   - `python tool08_ExtendedBaseline.py`
4. 运行主实验：
   - `python test12_RL_RLDSE_MAESTRO.py`

6、最近实验结论

- 在 VGG16 + EDP 上，新增指标后结果有小幅改善
- 当前更像是 `throughput` 起了主要作用
- `lambda_ext` 不是全程偏低，而是前期低、后期升到约 `0.8`
- 新指标权重仍存在向单一指标集中的风险，后续需要继续观察
