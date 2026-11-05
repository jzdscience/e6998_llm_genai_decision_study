# StartLux9B 规模匹配对照

2026-11-05。用户已授权实施。本页与 `configs/size_control_protocol.json` 固定新的
`size-control-anli-v1`；V1/V2/V3 原源码、冻结政策、结果及报告不改。
本机软件/输入核验完成不代表公司 L40S 已执行。

## 要补齐的比较

在 V3 完整 ANLI 官方 test 3,200 条上新增 StartLux9B，复用已验收的 StartLux4B、
Qwen4B、Qwen9B 原始预测。因此可同时查看同家族4B→9B与相同9B规模的家族差异。
不扩展到新的任务×模型笛卡尔积，不训练分类头或重新选择路由。
主描述比较为 StartLux9B−Qwen9B，次描述比较为 StartLux9B−StartLux4B；全部 R1/R2/R3
分别报告。此前test已查看，本轮是回顾性固定配方对照，不声称新的未见确认或架构因果效应。

使用原 V3 的3,200个test ID（R1 1,000、R2 1,000、R3 1,200）和原提示/类别顺序。
原90个development输入只用于两次串行重复性smoke，共180次调用；不看准确率决定是否继续。
全部3,290条输入使用实际官方renderer/tokenizer做长度检查，超长直接停止，不截断。
旧prediction中的模型接口名称仍保留；只有核验原config/revision/输入后才赋予比较角色。

## 固定模型与运行

- 模型为 `startlux-models/StartLux-Decision-v1.1-9B-Instant`，revision
  `18a17f734966b213b2f00fa9c915e8b9af2a9b20`。
- 使用已有vLLM0.31.0、固定StartLux插件源码和CUDA工具链。BF16、单L40S、单模型驻留，
  GPU memory utilization0.8，max_model_len8192，串行预测、每题最多2次尝试、120秒超时。
- 保留 bidirectional、think_mode=never、none_check=false、allow_none=false，禁用prefix cache。
  不自动换后端、精度、checkpoint或显存比例；0.8是冻结配置，能否启动须真实GPU验收。
- 9B官方choice温度是 **1.1014**，不是原4B的1.1103。固定config、decision_config和权重
  index的SHA在协议中；下载后读取模型文件再次核验。权重index记录tensor字节总量
  19,306,216,416，不代表实际RAM/VRAM峰值。
- 旧环境和4B目录只读复用，新权重、cache、预算/session、进程元数据保存在独立输出根。
  只在新Slurm job ID且用户显式 `--new-allocation` 后建立新session，不延长同一allocation。
  自动读取当前squeue剩余时间并留5分钟；历史deadline不能当当前资源证据。
  初始化要求数值Slurm job ID，GPU预检要求恰好一张L40S。启动前只读核验compute进程为空，
  遇到已有GPU工作直接停止，不终止其他进程。完整结果重启时先验身份链，跳过新session初始化。

官方资料于11/5核验：[固定checkpoint文件](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant/tree/18a17f734966b213b2f00fa9c915e8b9af2a9b20)、
[固定decision配置](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant/blob/18a17f734966b213b2f00fa9c915e8b9af2a9b20/decision_config.json)、
[官方模型卡](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-9B-Instant)。
官方其他硬件速度不用于本项目性能结论。

## 证据与停止规则

1. 从**已完成**的V3根校验固定公开输入、manifest、原freeze/access/evaluation哈希及三组
   completion、token、smoke、raw response与原始prediction hash，保存只读引用副本。
   此步不打开旧sealed gold；准备数据目录本身不能代替已完成V3结果。
2. 当前真实GPU doctor与模型配置确定后，在正式预测前保存配置freeze。不拟合任何政策。
   每次启动做90题×2串行smoke，保存每次尝试；这验证重复性，不是独立后端正确性对照。
3. 每25条原子落盘，终止结果不覆盖；恢复时核验配置与原freeze，跳过已保存sample ID。
   完整completion必须绑定所有预测、原始响应、exact-token记录与smoke。正式错误计错，
   沿用已有运行错误率停止规则，不能按准确率选择checkpoint、轮次或重跑。
4. 完整预测检查通过后写首次evaluation-access，再读取按原SHA固定的已见gold。访问收据
   已有但最终评价缺失时停止供审查，不静默重算；已完成评价只允许完全一致的重放。
5. 报告pooled accuracy、失败数、每round结果及两组配对救回/误伤计数；2,000次固定seed6998
   normalized-premise cluster bootstrap提供描述性区间。跨round同premise共同抽样。
   同步复算历史三模型direct指标并与原V3评价对齐。不宣称等效、因果效应或新路由收益。
6. 导出 `size_control_bundle.zip` 保留引用预测、配置/冻结/访问/评价、运行日志及诊断，排除
   权重和环境。部分包明确只是中间状态；只有 `SIZE_CONTROL_COMPLETE` 表示流程结束。

该实验补质量规模对照；单模型串行采集时间不代替HTTP服务吞吐、共驻留级联或SLO测量。
新增CPU/GPU服务比较采用另外的独立协议与输出目录。
