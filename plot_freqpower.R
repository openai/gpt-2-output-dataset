require(ggplot2)
require(data.table)

##
# Estimate NLL on webtext.test using 4 models
d_webtext_gpt2 <- fread("plot/webtext.test.model=gpt2.freq_power.csv")
d_webtext_medium <- fread("plot/webtext.test.model=gpt2-medium.freq_power.csv")
d_webtext_large <- fread("plot/webtext.test.model=gpt2-large.freq_power.csv")
d_webtext_xl <- fread("plot/webtext.test.model=gpt2-xl.freq_power.csv")

d_webtext_gpt2$model = "gpt2"
d_webtext_medium$model = "gpt2-medium"
d_webtext_large$model = "gpt2-large"
d_webtext_xl$model = "gpt2-xl"

d_webtext_c <- rbindlist(list(d_webtext_gpt2, d_webtext_medium, d_webtext_large, d_webtext_xl))
p_webtext_c <- ggplot(d_webtext_c, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("plot/webtext.test.4models.freq_power.pdf", plot=p_webtext_c)

# small-117M.test
d_small_gpt2 <- fread("plot/small-117M.test.model=gpt2.freq_power.csv")
d_small_medium <- fread("plot/small-117M.test.model=gpt2-medium.freq_power.csv")
d_small_large <- fread("plot/small-117M.test.model=gpt2-large.freq_power.csv")
d_small_xl <- fread("plot/small-117M.test.model=gpt2-xl.freq_power.csv")

d_small_gpt2$model = "gpt2"
d_small_medium$model = "gpt2-medium"
d_small_large$model = "gpt2-large"
d_small_xl$model = "gpt2-xl"

d_small_c <- rbindlist(list(d_small_gpt2, d_small_medium, d_small_large, d_small_xl))
p_small_c <- ggplot(d_small_c, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("plot/small-117M.test.4models.freq_power.pdf", plot=p_small_c)

# small-117M-k40.test
d_small_k40_gpt2 <- fread("plot/small-117M-k40.test.model=gpt2.freq_power.csv")
d_small_k40_medium <- fread("plot/small-117M-k40.test.model=gpt2-medium.freq_power.csv")
d_small_k40_large <- fread("plot/small-117M-k40.test.model=gpt2-large.freq_power.csv")
d_small_k40_xl <- fread("plot/small-117M-k40.test.model=gpt2-xl.freq_power.csv")

d_small_k40_gpt2$model = "gpt2"
d_small_k40_medium$model = "gpt2-medium"
d_small_k40_large$model = "gpt2-large"
d_small_k40_xl$model = "gpt2-xl"

d_small_k40_c <- rbindlist(list(d_small_k40_gpt2, d_small_k40_medium, d_small_k40_large, d_small_k40_xl))
p_small_k40_c <- ggplot(d_small_k40_c, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("plot/small-117M-k40.test.4models.freq_power_1k.pdf", plot=p_small_k40_c)

# medium-345M.test
d_medium_gpt2 <- fread("plot/medium-345M.test.model=gpt2.freq_power_1k.csv")
d_medium_medium <- fread("plot/medium-345M.test.model=gpt2-medium.freq_power_1k.csv")
d_medium_large <- fread("plot/medium-345M.test.model=gpt2-large.freq_power_1k.csv")
d_medium_xl <- fread("plot/medium-345M.test.model=gpt2-xl.freq_power_1k.csv")

d_medium_gpt2$model = "gpt2"
d_medium_medium$model = "gpt2-medium"
d_medium_large$model = "gpt2-large"
d_medium_xl$model = "gpt2-xl"

d_medium_c <- rbindlist(list(d_medium_gpt2, d_medium_medium, d_medium_large, d_medium_xl))
p_medium_c <- ggplot(d_medium_c, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("plot/medium-345M.test.4models.freq_power_1k.pdf", plot=p_medium_c)

# large-762M.test

# large-762M-k40.test

# xl-1542M.test

# xl-1542M-k40.test


####
# power~freq plots for different datasets
d_gpt2_webtext <- d_webtext_gpt2[,.(freq, power)]
d_gpt2_webtext$data <- "webtext"

d_gpt2_small <- d_small_gpt2[,.(freq, power)]
d_gpt2_small$data <- "small"

d_gpt2_small40k <- d_small_k40_gpt2[,.(freq, power)]
d_gpt2_small40k$data <- "small-40k"

d_gpt2_webtext_small <- rbindlist(list(d_gpt2_webtext, d_gpt2_small, d_gpt2_small40k))
p_gpt2_webtext_small <- ggplot(d_gpt2_webtext_small, aes(freq, power)) +
  geom_smooth(aes(linetype = data, fill = data, colour = data))
ggsave("plot/gpt2_webtext_small_small40k.freq_power.pdf", plot=p_gpt2_webtext_small)