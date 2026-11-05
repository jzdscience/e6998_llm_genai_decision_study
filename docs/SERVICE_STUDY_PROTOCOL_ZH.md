# CLINC CPU/GPU 同口径服务实验

2026-11-05，`clinc-service-v1`。用户授权实施，协议固定在
[service_study_protocol.json](../configs/service_study_protocol.json)。这是新的独立服务实验，
不重写V1–V3或轻量基线的源码、模型、参数和原结果。本机验证不等于远端实测完成。

## 固定比较

| 项目 | 冻结设置 |
|---|---|
| 条件 | MiniLM400、MiniLM1600、StartLux4B、Qwen4B，均为直接服务，无新路由 |
| 输入 | 原CLINC main600（450银行、150OOS），同一ID与原文本/任务定义 |
| 到达 | Poisson；5、25、100请求/秒；每repeat/rate四条件共享相同到达和样本表 |
| 时长 | 每cell180秒，重复5次；60cells，合计3小时正式到达窗口 |
| 请求数 | 固定随机种子生成469,804次offered请求；重复请求不增加独立质量样本 |
| 顺序 | 每repeat预先打乱condition，condition内预先打乱rate；不按结果改顺序 |
| client | 16workers，最多64个active+queued；每offer只有1次HTTP尝试，超时30秒 |
| SLO | 从预定到达至观察到完成≤500ms，包括generator lag、client queue和HTTP |
| drain | 到达窗口后最多60秒；父进程另给5秒退出宽限，超限终止所属测量进程和服务 |
| 隔离 | 每个cell重新启动服务并串行预热64条dev输入；服务退出后再进入下一cell |
| CPU | 原冻结MiniLM与保存LR头；每请求实际tokenize/encode，无预测查表或embedding缓存；batch1，数值线程1 |
| GPU | 复用固定4B revision、BF16、原接口/提示、no-thinking；独立单L40S，显存份额0.4 |

三个速率是预设测试水平，不保证分别对应每个服务的轻载、临界和过载。若某条件在100/s
仍没有接近饱和，如实报告测试范围内未观察到拐点，不根据main结果临时添加速率。
启动、模型下载、预热、drain、导出均不属于3小时到达窗口，单独记录，不能保证总工期。

每cell重新启动是为了隔离超时后的后端工作。客户端HTTP超时或future结束不能证明GPU
计算已经停止，普通health响应也不能证明队列为空。本协议采用所属进程组停止后重新启动，
不把残留工作带入下一rate。所有正式cell保留，包括失败多、速度慢和拒绝多的cell。

## 公平比较的边界

同一工作负载通过各模型的真实接口处理。MiniLM输入原文本；GPU输入同一原文本与固定
任务/类别定义，这是部署整体比较，不是相同token数或架构因果对照。共同client上限
不代表backend调度相同；StartLux前端原max-inflight16与CPU单worker等实际配置单列。

MiniLM400/1600使用过任务标签，LLM条件是原固定提示配置。main数据已被查看，本轮质量
属于回顾性服务工作负载表现。64条development预热中60条银行输入已属于1600训练池，
只作为预热，不以其质量判断是否继续或选择模型。

CPU路径不进行GPU推理，但仍占用公司GPU allocation。记录CPU型号、affinity/cpuset、
cgroup、Slurm job/host、进程CPU ticks/RSS及GPU UUID/显存/利用率/功率原始样本。
单线程不等于独占一核；采样缺失明确表示不可用，不补零，不从这些数据虚构美元成本。

## 评价与证据

每offer保留sample ID、scheduled arrival、dispatch、worker start、completion、实际HTTP
响应/最终标签、错误与拒绝信息。client过载拒绝也进入offered分母；HTTP失败保留在admitted
时延分布中。报告p50/p95/p99、实际offered rate、及时成功goodput，以及**及时且正确goodput**。
正确性来自本次实际响应，不用旧准确率乘以吞吐量，也不从缓存预测代替模型调用。

prepared阶段保存600输入、64预热输入、15份共享schedule和60cell清单。collection不读取
V1 gold。所有60cell成功形成完整证据后freeze，写独立evaluation-access，再读取固定旧gold
计算质量。5次重复报告各次值、median/min/max，不能把469,804请求当独立样本计算极窄CI。
同repeat/rate是否跨allocation会明确标注；跨allocation不称同硬件配对比较。

completed记录绑定plan、traces、occupancy、resources，plan另绑定实际service与warmup。
恢复时重新核验固定完整清单、schedule逐offer覆盖、raw hashes和summary。已有完成cell
只复用不重跑；中断attempt保留，必须显式`--resume-interrupted`后新建attempt。watchdog
竞争产生的completion另存unaccepted，不能冒充正常完成。

原始结果写入独立`results/aux-clinc-service-v1`，导出`service_study_bundle.zip`。
权重、旧运行目录和环境不随包传输。入口见[两项联合运行说明](RUN_FOLLOWUPS_ZH.md)。
