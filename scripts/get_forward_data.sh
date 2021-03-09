EXEHOME=${PROJECTHOME}/codes/Diversifying-Dialogue-Generation-with-Non-Conversational-Text/src/forward
DATAHOME=${PROJECTHOME}/data/DailyDialogue/processed

cd ${EXEHOME}

python process.py ${DATAHOME} ${DATAHOME}