
## LangGraphパッケージを格納しておくためのステージを作成する
```sql
create stage package;
```

## LangGraphパッケージをSnowflakeにアップロードする
```bash
snow snowpark package create langgraph -c xxx
snow snowpark package upload -f langgraph.zip -c xxx -s package
```

## LangGraphパッケージを読み込んでSiSを実行する
- SiSで、LangGraphパッケージ「@package/langgraph.zip」を読み込みます
- requirements.txtの内容を、SiSのパッケージに反映します
- sis_langgraph_llm.pyをSiSにコピペし、実行します
