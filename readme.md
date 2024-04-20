Data
adjusted_code_to_pv: all stock record with adjuested stock price -> Dict<stock_code, stock_records>
code_to_pv: all stock record with non-adjuested stock price-> Dict<stock_code, stock_records>
stock_records: stock records  -> stock_record[]
stock_record: stock price and volume info
[MDATE,OPEN,HIGH,LOW,CLOSE,VOLUME,AMOUNT,ROI,TURNOVER,OUTSTANDING,MV,BID,OFFER,ROIB,MV%,AMT%,TRN_D,PER-TSE,PER-TEJ,PBR-TSE,PBR-TEJ,LIMIT,TEJ_PSR,DIV_YID,TEJ_CDIV,CLSCHG,HMLPCT,REFPRC,U_LIMIT,D_LIMIT,XATTN1,XATTN2,XSTAT1,PMKT]

如何更新價格日資料
1. 到“未調整股價(日)”並匯出資料
2. 移除掉第一行 (columns), 避免 decode 出錯
3. 執行 script/process_tej_daily_raw_data.py

如何更新 EPS 資料
1. 到 TEJ 選"IFRS以合併為主簡表(單季)-全產業"
2. 執行 script/process_tej_finance_data.py (記得更新檔案位置)

