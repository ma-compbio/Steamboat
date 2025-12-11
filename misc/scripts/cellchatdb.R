load("C:/Files/data/CellChat/CellChatDB.human.rda")
write.csv(CellChatDB.human$interaction, file="C:/Files/data/CellChat/CellChatDB_human_interaction.csv", row.names=FALSE)

load("C:/Files/data/CellChat/CellChatDB.mouse.rda")
write.csv(CellChatDB.mouse$interaction, file="C:/Files/data/CellChat/CellChatDB_mouse_interaction.csv", row.names=FALSE)