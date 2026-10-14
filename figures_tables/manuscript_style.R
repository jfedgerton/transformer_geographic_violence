# manuscript_style.R -- one colour and shape per model, used by every results figure (RColorBrewer Dark2).
# Source from each figure script; never redefine these values locally.
COL <- c(persistence = "#666666",   # gray
         base        = "#7570B3",   # purple  (gradient boosting)
         emb_base    = "#1B9E77",   # teal    (base + embedding = pretrained encoder)
         random      = "#D95F02",   # orange  (random encoder)
         scratch     = "#E7298A")   # pink    (trained from scratch)
SHP <- c(persistence = 16,   # circle
         base        = 18,   # diamond
         emb_base    = 15,   # square
         random      = 18,   # diamond (filled)
         scratch     = 17)   # triangle
EMB_LIGHT <- "#8FD1BD"       # lighter teal for secondary series (pooled, individual seeds)
LINE_GRAY <- "#D9D9D9"       # connector lines in dot plots
# shape 18 (diamond) draws smaller than 15/16/17 at the same size; scale it up so markers look equal
SZ <- c(persistence = 1, base = 1.35, emb_base = 1, random = 1.35, scratch = 1)
