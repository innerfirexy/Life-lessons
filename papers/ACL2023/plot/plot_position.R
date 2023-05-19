require("data.table")
require("ggplot2")

setwd("plot")

##
# Compressed gestures
data1 <- fread("trm_gesture_compressed_ppls_test.txt")
setnames(data1, c("dialogID", "ppl", "position"))
unique(data1$dialogID)
#  [1]  0  1  2  3  4  5  6  7  8  9 10

data2 <- fread("trm_gesture_compressed_ppls_train.txt")
setnames(data2, c("dialogID", "ppl", "position"))
unique(data2$dialogID)
max(data2$dialogID) # 41

data1[, dialogID := dialogID + max(data2$dialogID) + 1]
unique(data1$dialogID)
# [1] 42 43 44 45 46 47 48 49 50 51 52
data1$dialogID <- as.integer(data1$dialogID)

dt_compressed <- rbindlist(list(data1, data2))
# p <- ggplot(dt_compressed, aes(x=position, y=ppl)) + geom_smooth()
# ggsave("trm_gesture_compressed_ppl_all.pdf", plot = p, width = 5, height = 5)

dt_comp_trm <- copy(dt_compressed)
dt_comp_trm$type <- "compressed"


##
# Uncompressed gestures
data1 <- fread("trm_gesture_ppls_test.txt")
setnames(data1, c("dialogID", "ppl", "position"))
unique(data1$dialogID)
#  [1]  0  1  2  3  4  5  6  7  8  9

data2 <- fread("trm_gesture_ppls_train.txt")
setnames(data2, c("dialogID", "ppl", "position"))
unique(data2$dialogID)
max(data2$dialogID) # 42

data_raw <- rbindlist(list(data1, data2))
summary(data_raw$position)
# Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
# 0.00   16.00   33.00   43.22   57.00  265.00
sd(data_raw$position)
# 41.60697

# p2 <- ggplot(data_raw, aes(x=position, y=ppl)) + geom_smooth()
# ggsave("trm_gesture_ppl_all.pdf", plot = p2, width = 5, height = 5)

dt_raw_trm <- copy(data_raw)
dt_raw_trm$type <- "raw"

dt_trm <- rbindlist(list(dt_comp_trm, dt_raw_trm))
dt_trm$model <- "Transformer"


####
# LSTM data
####
# Compressed
data1 <- fread("lstm_gesture_compressed_ppls_test.txt")
setnames(data1, c("dialogID", "ppl", "position"))
unique(data1$dialogID)

data2 <- fread("lstm_gesture_compressed_ppls_train.txt")
setnames(data2, c("dialogID", "ppl", "position"))
unique(data2$dialogID)
max(data2$dialogID) # 41

data1[, dialogID := dialogID + max(data2$dialogID) + 1]
unique(data1$dialogID)
# [1] 42 43 44 45 46 47 48 49 50 51 52
data1$dialogID <- as.integer(data1$dialogID)

data_compressed <- rbindlist(list(data1, data2))
# p <- ggplot(data_compressed, aes(x=position, y=ppl)) + geom_smooth()
# ggsave("lstm_gesture_compressed_ppl_all.pdf", plot = p, width = 5, height = 5)

dt_comp_lstm <- copy(data_compressed)
dt_comp_lstm$type <- "compressed"


# Uncompressed
data1 <- fread("lstm_gesture_ppls_test.txt")
setnames(data1, c("dialogID", "ppl", "position"))
unique(data1$dialogID)

data2 <- fread("lstm_gesture_ppls_train.txt")
setnames(data2, c("dialogID", "ppl", "position"))
unique(data2$dialogID)
max(data2$dialogID) # 41

data1[, dialogID := dialogID + max(data2$dialogID) + 1]
unique(data1$dialogID)
# [1] 42 43 44 45 46 47 48 49 50 51 52
data1$dialogID <- as.integer(data1$dialogID)

data_raw <- rbindlist(list(data1, data2))
# p <- ggplot(data_raw, aes(x=position, y=ppl)) + geom_smooth()
# ggsave("lstm_gesture_ppl_all.pdf", plot = p, width = 5, height = 5)

dt_raw_lstm <- copy(data_raw)
dt_raw_lstm$type <- "raw"

dt_lstm <- rbindlist(list(dt_comp_lstm, dt_raw_lstm))
dt_lstm$model <- "LSTM"

# by gesture type
ppls_raw <- rbindlist(list(dt_lstm[type=="raw"], dt_trm[type=="raw"]))
ppls_comp <- rbindlist(list(dt_lstm[type=="compressed"], dt_trm[type=="compressed"]))

###
# Save and load
fwrite(dt_trm, "ppls_trm.txt")
fwrite(dt_lstm, "ppls_lstm.txt")
fwrite(ppls_raw, "ppls_raw.txt")
fwrite(ppls_comp, "ppls_comp.txt")

ppls_trm <- fread("ppls_trm.txt")
ppls_lstm <- fread("ppls_lstm.txt")
ppls_raw <- fread("ppls_raw.txt")
ppls_comp <- fread("ppls_comp.txt")


###
# Plot together
###
# Check the distribution of the `position` column
summary(ppls_raw$position)
  #  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  #  0.00   16.00   33.00   43.22   57.00  265.00 
mean(ppls_raw$position) # 43.22
sd(ppls_raw$position) # 41.61
# ratio of position <= 100
nrow(ppls_raw[position <= 100]) / nrow(ppls_raw) # 0.94

plot_ppls_raw <- ggplot(ppls_raw, aes(x=position, y=ppl)) +
  geom_smooth(method="gam", aes(color=model, fill=model, lty=model)) +
  annotate("rect", xmin=100, xmax=265, ymin=-0.5, ymax=3, alpha=0.2, fill="grey") +
  # scale_x_log10() + 
  scale_color_brewer(palette = "Set2") + scale_fill_brewer(palette = "Set2") +
  labs(x="Utterance position", y="Local entropy") +
  theme_bw() + theme(legend.position = c(0.2, 0.6))
ggsave("gesture_entropy_position.pdf", plot=plot_ppls_raw, width=5, height=5)

p_raw_part <- ggplot(ppls_raw[position <= 100], aes(x=position, y=ppl)) +
  geom_smooth(method="gam", aes(color=model, fill=model, lty=model)) +
  scale_color_brewer(palette = "Set2") + scale_fill_brewer(palette = "Set2") +
  labs(x="Utterance position", y="Local entropy") +
  theme_bw() + theme(legend.position = c(0.2, 0.9))
ggsave("gesture_entropy_position_part.pdf", plot=p_raw_part, width=5, height=5)


plot_ppls_comp <- ggplot(ppls_comp, aes(x=position, y=ppl)) +
  geom_smooth(method="gam", aes(color=model, fill=model, lty=model)) +
  scale_color_brewer(palette = "Set2") + scale_fill_brewer(palette = "Set2") +
  labs(x="Utterance position", y="Local entropy") +
  theme_bw() + theme(legend.position = c(0.8, 0.9))
ggsave("gesture_compressed_entropy_position.pdf", plot=plot_ppls_comp, width=5, height=5)


###
# Compare Transformer vs. LSTM
t.test(data_trm$ppl, data_lstm$ppl)

t.test(data_trm[type=="raw"]$ppl, data_lstm[type=="raw"]$ppl)

t.test(data_trm[type=="compressed"]$ppl, data_lstm[type=="compressed"]$ppl)

###
# Linear models for ppl vs. position increasing pattern
# LSTM
summary(lm(ppl ~ position, data = ppls_lstm[type=="raw" & position<=120]))
#              Estimate Std. Error t value Pr(>|t|)
# (Intercept) 2.3930072  0.0364196  65.707   <2e-16 ***
# position    0.0016616  0.0008112   2.048   0.0406 *

summary(lm(ppl ~ position, data = ppls_lstm[type=="compressed"]))
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)  4.67015    0.18008  25.933  < 2e-16 ***
# position     0.09661    0.01407   6.868 1.38e-11 ***

# Tramsformer
summary(lm(ppl ~ position, data = ppls_trm[type=="raw" & position<=120]))
# Estimate Std. Error t value Pr(>|t|)
# (Intercept) 2.507195   0.039102  64.119  < 2e-16 ***
# position    0.002276   0.000871   2.613  0.00902 **

summary(lm(ppl ~ position, data = ppls_trm[type=="compressed"]))
# Estimate Std. Error t value Pr(>|t|)
# (Intercept)  5.01426    0.19164  26.165  < 2e-16 ***
# position     0.09302    0.01497   6.214  8.6e-10 ***