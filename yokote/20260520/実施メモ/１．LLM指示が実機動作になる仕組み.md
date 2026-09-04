# DimOS G1/Go2: 自然言語指示が実機動作になる仕組み

作成日: 2026-05-20

## 要約

このリポジトリでは、G1/Go2 を「自然言語で操作できるロボット」として動かすために、DimOS の agentic blueprint を起動する。

大きな流れは次の通り。

```text
人間の入力
  -> /human_input
  -> McpClient の LLM agent
  -> MCP tool call
  -> @skill メソッド
  -> Navigation / WebRTC / cmd_vel / robot connection
  -> G1 または Go2 が動く
```

LLM が自由に任意コードを実行するのではなく、`@skill` として公開された関数だけを MCP tool として呼び出す構造になっている。

## 起動単位: agentic blueprint

### Go2

Go2 側の起動スクリプトは `dimensional-applications/run_robot.sh`。

```bash
dimos run unitree-go2-agentic
```

シミュレーションでは `dimensional-applications/run_sim.sh` が次を実行する。

```bash
dimos --simulation run unitree-go2-agentic
```

`unitree-go2-agentic` の中身は `dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic.py`。

```python
unitree_go2_agentic = autoconnect(
    unitree_go2_spatial,
    McpServer.blueprint(),
    McpClient.blueprint(),
    _common_agentic,
)
```

含まれる主なモジュール:

- `unitree_go2_spatial`: Go2 connection、navigation、mapping、spatial memory、perception
- `McpServer`: `@skill` を MCP tool として HTTP で公開
- `McpClient`: LLM agent。MCP tool を取得して、自然言語入力から tool call を作る
- `_common_agentic`: `NavigationSkillContainer`, `PersonFollowSkillContainer`, `UnitreeSkillContainer`, `WebInput`, `SpeakSkill`

### G1

G1 側の実機起動スクリプトは `dimensional-applications-g1/run_robot.sh`。

```bash
dimos run unitree-g1-agentic
```

このスクリプトは `.env` の `ROBOT_IP_G1` を読み、起動時だけ `ROBOT_IP` に代入してから DimOS を起動する。Go2 の `ROBOT_IP` と G1 の `ROBOT_IP_G1` を共存させるための薄いラッパーになっている。

シミュレーションでは `dimensional-applications-g1/run_sim.sh` が次を実行する。

```bash
dimos --simulation run unitree-g1-agentic-sim
```

G1 の agentic 構成は以下。

- `unitree-g1-agentic`: `unitree_g1` + `_agentic_skills`
- `unitree-g1-agentic-sim`: `unitree_g1_sim` + `_agentic_skills`

`_agentic_skills` は `dimos/robot/unitree/g1/blueprints/agentic/_agentic_skills.py` にあり、次を接続している。

```python
_agentic_skills = autoconnect(
    McpServer.blueprint(),
    McpClient.blueprint(system_prompt=G1_SYSTEM_PROMPT),
    NavigationSkillContainer.blueprint(),
    SpeakSkill.blueprint(),
    UnitreeG1SkillContainer.blueprint(),
)
```

G1 では `McpClient` に `G1_SYSTEM_PROMPT` を渡している。Go2 用のデフォルトプロンプトを使うと、Go2 固有の sport command を誤って選びやすいため。

## 入力経路

### `say.sh`

Go2/G1 ともに `say.sh` は `dimos agent-send` の薄いラッパー。

```bash
./say.sh "右手を上げて"
```

内部的には次を実行する。

```bash
dimos agent-send "右手を上げて"
```

### `dimos agent-send`

`dimos/robot/cli/dimos.py` の `agent-send` コマンドは、ローカルの MCP server に `agent_send` tool call を投げる。

`agent_send` は `dimos/agents/mcp/mcp_server.py` の `McpServer` に `@skill` として定義されている。

```python
@skill
def agent_send(self, message: str) -> str:
    transport = pLCMTransport("/human_input")
    transport.publish(message)
```

つまり `agent-send` の実体は:

```text
CLI
  -> http://localhost:9990/mcp
  -> McpServer.agent_send()
  -> LCM topic /human_input
```

### `humancli`

`humancli` は Textual TUI。`dimos/utils/cli/human/humancli.py` で:

- `/human_input` に人間の入力を publish
- `/agent` を購読して agent の返答や tool call を表示
- `/agent_idle` を購読して thinking 状態を表示

`agent-send` と違い、`humancli` は MCP の `agent_send` を経由せず、直接 LCM topic に流す。

## LLM agent 側の処理

`McpClient` は `dimos/agents/mcp/mcp_client.py` にある DimOS `Module`。

主な stream:

- `human_input: In[str]`
- `agent: Out[BaseMessage]`
- `agent_idle: Out[bool]`

起動時の流れ:

1. `McpClient.start()` が `/human_input` を subscribe する
2. `McpClient.on_system_modules()` が MCP server から tool 一覧を取得する
3. `create_agent(model="gpt-4o", tools=tools, system_prompt=...)` で LangGraph/LangChain agent を作る
4. 人間の入力を `HumanMessage` として queue に入れる
5. agent が返答や tool call を生成する
6. 生成された message は `/agent` に publish される
7. 処理中/待機中は `/agent_idle` に publish される

デフォルトモデルは `gpt-4o`。そのため通常は `OPENAI_API_KEY` が必要。

## MCP server と skill discovery

`McpServer` は `dimos/agents/mcp/mcp_server.py` にある。

起動すると FastAPI/uvicorn で MCP endpoint を立てる。

```text
http://localhost:9990/mcp
```

主な JSON-RPC method:

- `initialize`
- `tools/list`
- `tools/call`

skill discovery の仕組み:

1. `@skill` decorator は `@rpc` を付与し、`__skill__ = True` を付ける
2. 各 `Module.get_skills()` が `__skill__` 付きメソッドを探す
3. LangChain の `tool(...)` を使って args schema を作る
4. `McpServer.on_system_modules()` が全 module の skill を集める
5. `tools/list` で LLM client に tool schema を返す
6. `tools/call` で該当 skill を RPC 経由で呼び出す

重要な制約:

- skill の docstring は LLM が見る tool description になる
- 引数の型注釈が schema になる
- MCP server の tool map は `func_name` を key にしているため、skill 名の衝突には注意が必要

## Skill から robot への接続

### Go2 の主な skill

`dimos/robot/unitree/unitree_skill_container.py`

- `relative_move(forward, left, degrees)`
  - 現在位置を TF から取得
  - 相対目標 `PoseStamped` を作る
  - navigation module の `set_goal()` を呼ぶ
  - navigation/movement manager が `cmd_vel` を生成し、`GO2Connection` が実機に送る

- `wait(seconds)`
  - 指定秒数待つ

- `current_time()`
  - 現在時刻を返す

- `execute_sport_command(command_name)`
  - Go2 の sport command を Unitree WebRTC topic に publish
  - 例: `StandUp`, `Sit`, `Hello`, `FrontPounce`, `Backflip`, `RecoveryStand`

`dimos/robot/unitree/go2/connection.py`

- `observe()`
  - Go2 camera の最新 frame を返す

`dimos/agents/skills/person_follow.py`

- `follow_person(query, ...)`
  - camera 画像から対象人物を検出・追跡し、visual servoing で追従

- `stop_following()`
  - 追従停止、`cmd_vel` に zero を publish

### G1 の主な skill

`dimos/robot/unitree/g1/skill_container.py`

- `move(x, y, yaw, duration)`
  - `Twist(linear=(x,y,0), angular=(0,0,yaw))` を作る
  - `G1Connection.move()` または `G1SimConnection.move()` を呼ぶ

- `execute_arm_command(command_name)`
  - `rt/api/arm/request` に WebRTC request を送る
  - 例: `Handshake`, `HighFive`, `Hug`, `HighWave`, `Clap`, `RightHandUp`, `ArmHeart`

- `execute_mode_command(command_name)`
  - `rt/api/sport/request` に WebRTC request を送る
  - 例: `WalkMode`, `WalkControlWaist`, `RunMode`

### 共通 skill

`dimos/agents/skills/navigation.py`

- `tag_location(location_name)`
  - 現在位置を semantic/spatial memory に名前付き保存

- `navigate_with_text(query)`
  - まず tagged location を検索
  - 次に現在カメラ画像から対象物を探す
  - 最後に semantic map を検索
  - 見つかれば navigation goal を設定

- `stop_navigation()`
  - navigation goal をキャンセル

`dimos/agents/skills/speak_skill.py`

- `speak(text, blocking=True)`
  - OpenAI TTS + audio output で発話する

## 実機 connection

### Go2

`GO2Connection` は `dimos/robot/unitree/go2/connection.py`。

主な役割:

- Unitree WebRTC connection を開始
- lidar / odom / video stream を DimOS stream に publish
- `cmd_vel` を subscribe して `connection.move(twist, duration)` を呼ぶ
- 起動時に standup、balance stand、obstacle avoidance 設定を実行
- 停止時に lie down する

Go2 の基本 stack は `dimos/robot/unitree/go2/blueprints/basic/unitree_go2_basic.py`。ここで visualization や transport も接続される。

### G1

実機 G1 は `dimos/robot/unitree/g1/connection.py` の `G1Connection`。

主な役割:

- `UnitreeWebRTCConnection(self.config.ip)` を開始
- `cmd_vel` を subscribe
- `move(twist, duration)` で WebRTC connection に速度指令を渡す
- `publish_request(topic, data)` で arm/mode command を送る

シミュレーション G1 は `dimos/robot/unitree/g1/mujoco_sim.py` の `G1SimConnection`。

主な役割:

- `MujocoConnection` を開始
- sim odom / lidar / video を publish
- `cmd_vel` を受けて MuJoCo 内の G1 を動かす

## 例: G1 に「右手を上げて」と言う場合

```text
./say.sh "右手を上げて"
  -> dimos agent-send "右手を上げて"
  -> McpServer.agent_send(message)
  -> /human_input
  -> McpClient が受信
  -> G1_SYSTEM_PROMPT + tool schema を見て LLM が判断
  -> execute_arm_command(command_name="RightHandUp") を tool call
  -> McpServer tools/call
  -> UnitreeG1SkillContainer.execute_arm_command()
  -> G1Connection.publish_request("rt/api/arm/request", {"api_id": 7106, "parameter": {"data": 23}})
  -> G1 が右手を上げる
```

## 例: Go2 に「少し前に進んで」と言う場合

```text
./say.sh "少し前に進んで"
  -> dimos agent-send
  -> /human_input
  -> McpClient が受信
  -> Go2 SYSTEM_PROMPT + tool schema を見て LLM が判断
  -> relative_move(forward=..., left=..., degrees=...) などを tool call
  -> UnitreeSkillContainer.relative_move()
  -> NavigationInterfaceSpec.set_goal()
  -> navigation / movement manager
  -> cmd_vel
  -> GO2Connection.move()
  -> Unitree WebRTC
  -> Go2 が移動
```

## 確認した主なファイル

- `dimensional-applications/run_robot.sh`
- `dimensional-applications/run_sim.sh`
- `dimensional-applications/say.sh`
- `dimensional-applications-g1/run_robot.sh`
- `dimensional-applications-g1/run_sim.sh`
- `dimensional-applications-g1/say.sh`
- `dimos/robot/unitree/go2/blueprints/agentic/unitree_go2_agentic.py`
- `dimos/robot/unitree/go2/blueprints/agentic/_common_agentic.py`
- `dimos/robot/unitree/g1/blueprints/agentic/_agentic_skills.py`
- `dimos/robot/unitree/g1/blueprints/agentic/unitree_g1_agentic.py`
- `dimos/robot/unitree/g1/blueprints/agentic/unitree_g1_agentic_sim.py`
- `dimos/agents/mcp/mcp_client.py`
- `dimos/agents/mcp/mcp_server.py`
- `dimos/agents/annotation.py`
- `dimos/core/module.py`
- `dimos/robot/unitree/unitree_skill_container.py`
- `dimos/robot/unitree/g1/skill_container.py`
- `dimos/robot/unitree/go2/connection.py`
- `dimos/robot/unitree/g1/connection.py`
- `dimos/robot/unitree/g1/mujoco_sim.py`
- `dimos/agents/skills/navigation.py`
- `dimos/agents/skills/speak_skill.py`
- `dimos/agents/skills/person_follow.py`
- `dimos/agents/system_prompt.py`
- `dimos/robot/unitree/g1/system_prompt.py`
