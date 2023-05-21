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
# Top 6 tokens
#    gesture  ppl_mean    ppl_sd count  ci_upper  ci_lower
# 1:      63  2.968982  12.40563 42367  3.093411  2.853379
# 2:      64  6.057499  30.75231 20540  6.523445  5.664553
# 3:      56  6.252668  44.32664 20354  6.893801  5.644288
# 4:       0 12.248335  86.93096 17003 13.643883 10.843682
# 5:      72 13.989324  75.63458  9264 15.702493 12.596356
# 6:      36 51.578032 326.96861  2762 65.384295 40.977287

# Map the old gesture IDs to new ones
# Old: L x R => New: (L-1) * 9 + R   (See Eq. 2 in the paper)
# 63 = 9 x 7 => (9-1) * 9 + 7 = 79
# 64 = 8 x 8 => (8-1) * 9 + 8 = 71
# 56 = 8 x 7 => (8-1) * 9 + 7 = 70
# 0 => none gesture
# 72 = 9 x 8 => (9-1) * 9 + 8 = 80
# 36 = 9 x 4 => (9-1) * 9 + 4 = 76
# Thus, the top-5 most common gesture tokens are 79, 71, 70, 80, and 76.

# Select top 6 tokens and conver to new tokens
d.top6 <- data_raw_summ[1:6,]
d.top6$gesture <- as.character(c(79, 71, 70, 0, 80, 76))

# Add proportion column
total_count <- sum(data_raw_summ$count) # 121540 ~ same as # of word tokens
d.top6$proportion <- d.top6$count / total_count

# Save to csv
fwrite(d.top6[, c("gesture", "count", "proportion", "ppl_mean")], 
  "top6_stats.csv", sep=",", quote=FALSE, row.names=FALSE)


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
data_raw_summ$rank <- seq(1:nrow(data_raw_summ))
data_comp_summ$rank <- seq(1:nrow(data_comp_summ))

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
data_raw_summ[gesture == "63", gesture_new := "79"]
data_raw_summ[gesture == "56", gesture_new := "70"]
data_raw_summ[gesture == "64", gesture_new := "71"]
data_comp_summ[gesture == "63", gesture_new := "79"]
data_comp_summ[gesture == "56", gesture_new := "70"]
data_comp_summ[gesture == "64", gesture_new := "71"]

# Get coordinates for annotation
data_raw_summ[gesture_new %in% c("79", "70", "71"), .(rank, count)]
#    rank count
# 1:    1 42367
# 2:    2 20540
# 3:    3 20354
data_comp_summ[gesture_new %in% c("79", "70", "71"), .(rank, count)]
#    rank count
# 1:    1  6430
# 2:    2  5625
# 3:    3  4944
data_freq[rank<=3, .(mean_count = mean(count)), by=.(rank)]
#    rank mean_count
# 1:    1    24398.5
# 2:    2    13082.5
# 3:    3    12649.0
# The `mean_count` is used to determine the y coordinate of the annotation


p_freq <- ggplot(data_freq, aes(x=rank, y=count)) +
  # geom_col(data = data_freq[rank <= 3,], 
  #   aes(fill=type), alpha=0.5, position = position_dodge(), width=0.2) +
  geom_smooth(aes(color=type, lty=type), se=FALSE, linewidth = 1) +
  geom_point(data = data_freq, aes(color=type, shape=type), size=3) +
  annotate("label", 
            # x=c(0, 0.693, 1.099), y=c(9.67, 9.13, 8.89),
            x=c(1,2,3), y=c(24398.5, 13082.5, 12649.0),
           label=c("<79>", "<70>", "<71>"), hjust="inward") +
  scale_color_brewer("Gesture token type", palette = "Set1") +
  scale_fill_brewer("Gesture token type", palette = "Set1") +
  scale_linetype_discrete("Gesture token type") +
  scale_shape_discrete("Gesture token type") +
  scale_x_log10() + scale_y_log10() +
  labs(x="Rank of gesture token (log scaled)", y="Frequency count (log scaled)") +
  theme_bw() +
  theme(
    # axis.text.x = element_blank(), 
    # axis.ticks.x = element_blank(),
    panel.grid.major = element_blank(), legend.position = c(.8,.8))
ggsave("freq_rank_log-log.pdf", plot = p_freq, width = 6, height = 4)
