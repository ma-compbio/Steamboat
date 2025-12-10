load("C:/Files/data/CellChat/CellChatDB.human.rda")
write.csv(CellChatDB.human$interaction, file="C:/Files/data/CellChat/CellChatDB_human_interaction.csv", row.names=FALSE)
