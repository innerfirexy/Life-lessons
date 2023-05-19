require("data.table")
require("ggplot2")

setwd("papers/ACL2023/plot")

###
# Uncompressed gestures
data_raw <- fread("trm_gesture_gesture_ppl_per_token.txt")
setkey(data_raw, gesture)
data_raw$gesture <- as.character(data_raw$gesture)

data_raw_summ <- data_raw[,
  .(ppl_mean = mean(ppl), ppl_sd = sd(ppl), count = .N, ci_upper = mean_cl_boot(ppl)$ymax, ci_lower = mean_cl_boot(ppl)$ymin),
  by=gesture]
data_raw_summ <- data_raw_summ[order(-count)]
# top 5
# 31	63	2.96898	12.4056	42367	3.09417	2.86052
# 32	64	6.05750	30.7523	20540	6.49515	5.65289
# 30	56	6.25267	44.3266	20354	6.86402	5.66097
# 1	    0	12.2483	86.9310	17003	13.4979	10.9667
# 33	72	13.9893	75.6346	9264	15.5515	12.5498
# 6	    36	51.5780	326.969	2762	64.6469	41.1306
# 7	    32	70.4669	439.890	2038	91.4624	53.6883
# 8	    24	142.672	688.943	1035	182.851	105.810
# 9	    42	122.769	457.413	796	    163.085	95.5743
# 10	40	212.412	858.591	718	    282.316	159.825

# Map the old gesture IDs to new ones
# Old: L x R => New: (L-1) * 9 + R   (See Eq. 2 in the paper)
# 63 = 9 x 7 => (9-1) * 9 + 7 = 79
# 64 = 8 x 8 => (8-1) * 9 + 8 = 71
# 56 = 8 x 7 => (8-1) * 9 + 7 = 70
# 0 => none gesture
# 72 = 9 x 8 => (9-1) * 9 + 8 = 80
# 36 = 9 x 4 => (9-1) * 9 + 4 = 76
# Thus, the top-5 most common gesture tokens are 79, 71, 70, 80, and 71.


sum(data_raw_summ$count)
# 121540 ~ same as # of word tokens

sorted_gestures <- data_raw_summ$gesture
data_raw_summ$gesture <- factor(data_raw_summ$gesture, levels=sorted_gestures)

p_raw <- ggplot(data_raw_summ[1:5,], aes(x=gesture, y=ppl_mean)) +
  geom_linerange(aes(ymin=ci_lower, ymax=ci_upper))


###
# Compressed gesture
data_comp <- fread("trm_gesture_gesture_compressed_ppl_per_token.txt")
setkey(data_comp, gesture)
data_comp$gesture <- as.character(data_comp$gesture)

data_comp_summ <- data_comp[,
  .(ppl_mean = mean(ppl), ppl_sd = sd(ppl), count = .N, ci_upper = mean_cl_boot(ppl)$ymax, ci_lower = mean_cl_boot(ppl)$ymin),
  by=gesture]
data_comp_summ <- data_comp_summ[order(-count)]

head(data_comp_summ, 10)
# gesture  ppl_mean     ppl_sd count  ci_upper  ci_lower
# 1:      63  2.779571   2.861976  6430  2.849425  2.713113
# 2:      56  4.248089   4.570749  5625  4.362672  4.128994
# 3:      64  3.799365   3.543830  4944  3.892387  3.704802
# 4:      72  7.308519  13.058217  3041  7.780707  6.869698
# 5:      36 19.870917  32.763101  1164 21.800064 18.237990
# 6:      32 29.033029  35.489065   984 31.417867 27.036412
# 7:       0 57.985044  95.309595   482 67.389461 50.423464
# 8:      24 67.330075  66.252950   431 74.038036 61.241462
# 9:      42 87.113518  74.324703   409 94.280450 80.454942
sum(data_comp_summ$count)
# 26120 ~ smaller than raw count


###
# Frequency vs. Rank plot
data_raw_summ$gesture <- factor(data_raw_summ$gesture, levels=as.character(data_raw_summ))
data_raw_summ$rank <- seq(1:nrow(data_raw_summ))

p_raw <- ggplot(data_raw_summ, aes(x=log(rank), y=log(count))) +
  geom_col(fill="steelblue", alpha=0.5) +
  #geom_point() +
  geom_smooth(aes(group=1), se=FALSE, size = 1) +
  # annotate("label", x=c(0, 0.693, 1.099), y=c(9.67, 9.13, 7.89),
  #          label=c("<70>", "<69>", "<78>"), hjust="inward") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), panel.grid.major = element_blank())


data_comp_summ$gesture <- factor(data_comp_summ$gesture, levels=as.character(data_comp_summ))
data_comp_summ$rank <- seq(1:nrow(data_comp_summ))

p_comp <- ggplot(data_comp_summ, aes(x=log(rank), y=log(count))) +
  geom_col(fill="#FC4E07", alpha=0.5) +
  geom_smooth(aes(group=1), se=FALSE, color = "firebrick1", size = 1) +
  # annotate("label", x=c(0, 0.693, 1.099), y=c(9.67, 9.13, 7.89),
  #          label=c("<70>", "<69>", "<78>"), hjust="inward") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(), panel.grid.major = element_blank())

# Combined
data_raw_summ$type <- "Raw"
data_comp_summ$type <- "Compressed"
data_freq <- rbindlist(list(
    data_raw_summ[,.(rank, count, type)], data_comp_summ[,.(rank, count, type)]
))
# Gesture token old => new
# 63 => 79
# 56 => 70
# 64 => 71
p_freq <- ggplot(data_freq, aes(x=log(rank), y=log(count))) +
  geom_col(aes(fill=type), alpha=0.5, position = position_dodge(), width=0.1) +
  geom_smooth(aes(color=type, lty=type), se=FALSE, linewidth = 1.5) +
  annotate("label", x=c(0, 0.693, 1.099), y=c(9.67, 9.13, 8.89),
           label=c("<79>", "<70>", "<71>"), hjust="inward") +
  scale_fill_manual("Gesture token type", values = c("#D55E00", "#0072B2")) + #E69F00 or #FC4E07
  scale_color_manual("Gesture token type", values = c("#D55E00", "#0072B2")) + #0072B2 or #FC4E07
  scale_linetype_discrete("Gesture token type") +
  labs(x="Rank of gesture token (log scaled)", y="Frequency count (log scaled)") +
  theme_bw() +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
        panel.grid.major = element_blank(), legend.position = c(.8,.8))
ggsave("freq_rank_log-log.pdf", plot = p_freq, width = 6, height = 4)