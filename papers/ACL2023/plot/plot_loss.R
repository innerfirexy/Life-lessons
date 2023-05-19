require(ggplot2)
require(data.table)

setwd("../data/")

# LSTM, mix model
dt1 = fread("lstm_mix_log.txt")
dt1$Input = 'mix (word + gesture)'

p1 <- ggplot(dt1[epoch<=15], aes(epoch, loss)) +
  geom_line() + geom_point()

# LSTM word model
dt2 = fread("lstm_word_log.txt")
dt2$Input = 'word'

dt_lstm <- rbindlist(list(dt1, dt2))
dt_lstm$model <- "LSTM"

p_lstm <- ggplot(dt_lstm[epoch<=15], aes(epoch, loss)) +
  geom_line(aes(color=Input, linetype=Input)) +
  geom_point(aes(shape=Input, color=Input)) +
  theme_bw() +
  theme(legend.position = c(0.75,0.85)) +
  scale_color_brewer(palette = "Set1") +
  labs(x="Epoch", y="Validation loss")
ggsave("loss_epoch_LSTM.pdf", plot=p_lstm, width = 4, height=4)


# Transformer, mix model
dt3 <- fread("trm_mix_log.txt")
dt3$Input <- "mix (word + gesture)"

dt4 <- fread("trm_word_log.txt")
dt4$Input <- "word"

dt_trm <- rbindlist(list(dt3, dt4))
dt_trm$model <- "Transformer"

p_trm <- ggplot(dt_trm[epoch<=15], aes(epoch, loss)) +
  geom_line(aes(color=Input, linetype=Input)) +
  geom_point(aes(color=Input, shape=Input)) +
  theme_bw() +
  theme(legend.position = c(0.75,0.85)) +
  scale_color_brewer(palette = "Set1") +
  labs(x="Epoch", y="Validation loss")
ggsave("loss_epoch_Trm.pdf", plot=p_trm, width = 4, height=4)


# Combine
dt_all <- rbindlist(list(dt_lstm, dt_trm))


## sum, concat, bilinear plot
dt_sum <- fread("lstm_mix_log.txt")
dt_sum$Method <- "sum"

dt_concat <- fread("lstm_mix_log_concat.txt")
dt_concat$Method <- "concat"

dt_bilinear <- fread("lstm_mix_log_bilinear.txt")
dt_bilinear$Method <- "bilinear"

dt_method_lstm <- rbindlist(list(dt_sum, dt_concat, dt_bilinear))

p_method_lstm <- ggplot(dt_method_lstm[epoch<=15], aes(epoch, loss)) +
  geom_line(aes(color=Method, linetype=Method)) +
  geom_point(aes(shape=Method, color=Method)) +
  theme_bw() +
  theme(legend.position = c(0.75,0.8)) +
  scale_color_brewer(palette = "Set1") +
  labs(x="Epoch", y="Validation loss")
ggsave("loss_epoch_LSTM_mix_method.pdf", plot=p_method_lstm, width = 4, height=4)


dt_sum <- fread("trm_mix_log.txt")
dt_sum$Method <- "sum"

dt_concat <- fread("trm_mix_log_concat.txt")
dt_concat$Method <- "concat"

dt_bilinear <- fread("trm_mix_log_bilinear.txt")
dt_bilinear$Method <- "bilinear"

dt_method_trm <- rbindlist(list(dt_sum, dt_concat, dt_bilinear))
p_method_trm <- ggplot(dt_method_trm[epoch<=15], aes(epoch, loss)) +
  geom_line(aes(color=Method, linetype=Method)) +
  geom_point(aes(shape=Method, color=Method)) +
  theme_bw() +
  theme(legend.position = c(0.75,0.8)) +
  scale_color_brewer(palette = "Set1") +
  labs(x="Epoch", y="Validation loss")
ggsave("loss_epoch_Trm_mix_method.pdf", plot=p_method_trm, width = 4, height=4)