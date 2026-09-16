# harvest グラフ・エディタ

`dimos/robot/unitree/g1/harvest/graph.py` の `build_harvest_graph()` が組み立てる
LangGraph `StateGraph`（12ノード・固定エッジ・4箇所の条件分岐 `add_conditional_edges`）
の**配線（ノード配置・エッジ・条件分岐マッピング）だけ**を GUI で可視化・編集し、
編集結果から `StateGraph` 構築コードを生成してエクスポートするツール。

バックエンドは FastAPI + uvicorn、フロントエンドは React Flow を使った単一 HTML
ファイル（ビルドステップ無し、ブラウザ上で esm.sh 経由の ESM import のみで動く）。

## 起動方法

```bash
.venv/bin/python -m dimos.robot.unitree.g1.harvest.graph_editor
# ブラウザで http://127.0.0.1:8420 を開く
```

オプション:

```bash
.venv/bin/python -m dimos.robot.unitree.g1.harvest.graph_editor --port 9000 --host 0.0.0.0 --reload
```

`fastapi` / `uvicorn[standard]` が `.venv` に無い場合は以下で追加する
（このプロジェクトは uv 管理、`.venv/bin/python -m pip` は使えない）:

```bash
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]"
```

## 使い方

1. ブラウザでページを開くと `GET /api/graph` が呼ばれ、現行の harvest グラフ
   （`DummyHarvestSkills` で組み立てたもの）がそのまま描画される。
2. ノードをドラッグして配置を変更できる。
3. **ノードの追加**: ツールバーの「＋ ノード追加」ボタン、またはキャンバスの
   何もないところをダブルクリックして名前を入力する。
4. **ノードのリネーム**: ノードをダブルクリックして新しい名前を入力する
   （そのノードを参照している既存エッジの `source`/`target` も自動で追従する）。
5. **ノードの削除**: ノードを選択して Delete / Backspace キー、またはサイド
   パネルの「このノードを削除」ボタン。`START`/`END`（開始・終了の擬似ノード）
   は削除・リネームできない。
6. **エッジの追加**: ノードのハンドル（縁）からドラッグして別のノードに接続する。
7. **エッジの削除**: エッジを選択して Delete / Backspace キー、またはサイド
   パネルの「このエッジを削除」ボタン。
8. **条件分岐マッピングの編集**: エッジをクリックして選択すると右のパネルに
   「条件分岐エッジにする」チェックボックスと「ラベル」入力欄が出る。
   チェックを入れると、そのエッジは `add_conditional_edges` のマッピング
   （ルーターがラベルの文字列を返したらこのエッジの行き先に遷移する、という
   対応）として扱われる。チェックが無い通常のエッジは固定の `add_edge`。
9. 「Python コードを生成」ボタンを押すと、現在の nodes/edges が
   `POST /api/export` に送られ、`StateGraph` 構築コードがサイドパネルの
   テキストエリアに表示される（読み取り専用・コピー可能）。

## スコープ外（このGUIではできないこと）

- **ノードの中身（Python実装）を書くことはできない。** ノードは「既存の関数名
  への参照」として扱われるだけ。既存ノード（`detect`/`select`/`grasp`/…)を
  そのまま使う場合、生成コードはその関数がどこかのスコープに既に存在する
  （元の `build_harvest_graph()` 内のクロージャ、など）ことを前提にした
  `g.add_node("detect", detect)` のような参照だけを出す。
  新規に追加したノード（またはリネームしたノード）には
  `def <name>(state: HarvestState) -> HarvestState:  # TODO: implement` という
  プレースホルダー関数を生成するので、実装は生成後に自分で書くこと。
- **ルーター関数（`route_after_select` 等)の判定ロジック自体は編集できない。**
  GUIが編集するのは「ルーターがどの文字列を返したらどのノードに飛ぶか」という
  マッピング表（`add_conditional_edges` の第3引数の dict）だけ。生成される
  ルーター関数は常に `raise NotImplementedError` のプレースホルダーで、
  判定ロジックは元の `route_after_*` 関数を参考に人手で実装する必要がある。
- **双方向同期・自動保存は無い。** 生成されたコードは画面に表示されるだけで、
  `graph.py` に自動で書き戻されることは無い。レビューした上で手動で反映すること。
- グラフ内にサイクル（`detect` ⇄ `select` など）があるため、初期表示の自動
  レイアウトは簡易的な段組みであり、完全に見やすい配置にはならない。ノードは
  手動でドラッグして整理すること。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `introspect.py` | `build_harvest_graph()` を組み立てて `CompiledStateGraph.get_graph()` を React Flow 用 JSON に変換する |
| `codegen.py` | 編集後の nodes/edges JSON から `StateGraph` 構築コードの文字列を生成する |
| `server.py` | FastAPI アプリ本体（`GET /api/graph`、`POST /api/export`、静的ファイル配信） |
| `static/index.html` | React Flow を使った単一ページ GUI |
| `__main__.py` | `python -m ...graph_editor` のエントリポイント（uvicorn 起動） |

## API

- `GET /api/graph` → `{"nodes": [{"id","label","position"}, ...], "edges": [{"id","source","target","conditional","label"}, ...]}`
- `POST /api/export`（body は上と同じ nodes/edges 形式）→ `{"code": "..."}`
