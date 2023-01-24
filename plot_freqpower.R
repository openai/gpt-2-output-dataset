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
d_medium_gpt2 <- fread("plot/medium-345M.test.model=gpt2.freq_power.csv")
d_medium_medium <- fread("plot/medium-345M.test.model=gpt2-medium.freq_power.csv")
d_medium_large <- fread("plot/medium-345M.test.model=gpt2-large.freq_power.csv")
d_medium_xl <- fread("plot/medium-345M.test.model=gpt2-xl.freq_power.csv")

d_medium_gpt2$model = "gpt2"
d_medium_medium$model = "gpt2-medium"
d_medium_large$model = "gpt2-large"
d_medium_xl$model = "gpt2-xl"

d_medium_c <- rbindlist(list(d_medium_gpt2, d_medium_medium, d_medium_large, d_medium_xl))
p_medium_c <- ggplot(d_medium_c, aes(freq, power)) +
  geom_smooth(aes(linetype = model, fill = model, colour = model))
ggsave("plot/medium-345M.test.4models.freq_power_1k.pdf", plot=p_medium_c)

# medium-345M-k40.test
d_medium_k40_gpt2 <- fread("plot/medium-345M-k40.test.model=gpt2.freq_power.csv")

# large-762M.test
d_large_gpt2 <- fread("plot/large-762M.test.model=gpt2.freq_power.csv")

# large-762M-k40.test
d_large_k40_gpt2 <- fread("plot/large-762M-k40.test.model=gpt2.freq_power.csv")

# xl-1542M.test
d_xl_gpt2 <- fread("plot/xl-1542M.test.model=gpt2.freq_power.csv")

# xl-1542M-k40.test
d_xl_k40_gpt2 <- fread("plot/xl-1542M-k40.test.model=gpt2.freq_power.csv")


####
# power~freq plots for different datasets
####
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

# medium
d_gpt2_medium <- d_medium_gpt2[,.(freq, power)]
d_gpt2_medium$data <- "medium"
d_gpt2_medium40k <- d_medium_k40_gpt2[,.(freq, power)]
d_gpt2_medium40k$data <- "medium-40k"

d_gpt2_wb_sm_me <- rbindlist(list(d_gpt2_webtext, d_gpt2_small, d_gpt2_medium))
p_gpt2_wb_sm_me <- ggplot(d_gpt2_wb_sm_me, aes(freq, power)) +
  geom_smooth(aes(linetype = data, fill = data, colour = data))
ggsave("plot/gpt2_webtext_small_medium.freq_power.pdf", plot=p_gpt2_wb_sm_me)

d_gpt2_wb_me_me40k <- rbindlist(list(d_gpt2_webtext, d_gpt2_medium, d_gpt2_medium40k))
p_gpt2_wb_me_me40k <- ggplot(d_gpt2_wb_me_me40k, aes(freq, power)) +
  geom_smooth(aes(linetype = data, fill = data, colour = data))
ggsave("plot/gpt2_webtext_medium_medium40k.freq_power.pdf", plot=p_gpt2_wb_me_me40k)

# large
d_gpt2_large <- d_large_gpt2[,.(freq, power)]
d_gpt2_large$data <- "large"
d_gpt2_large40k <- d_large_k40_gpt2[,.(freq, power)]
d_gpt2_large40k$data <- "large-40k"

d_gpt2_wb_la_la40k <- rbindlist(list(d_gpt2_webtext, d_gpt2_large, d_gpt2_large40k))
p_gpt2_wb_la_la40k <- ggplot(d_gpt2_wb_la_la40k, aes(freq, power)) +
  geom_smooth(aes(linetype = data, fill = data, colour = data))
ggsave("plot/gpt2_webtext_large_large40k.freq_power.pdf", plot=p_gpt2_wb_la_la40k)

# xl
d_gpt2_xl <- d_xl_gpt2[,.(freq, power)]
d_gpt2_xl$data <- "xl"
d_gpt2_xl40k <- d_xl_k40_gpt2[,.(freq, power)]
d_gpt2_xl40k$data <- "xl-40k"

d_gpt2_wb_xl_xl40k <- rbindlist(list(d_gpt2_webtext, d_gpt2_xl, d_gpt2_xl40k))
p_gpt2_wb_xl_xl40k <- ggplot(d_gpt2_wb_xl_xl40k, aes(freq, power)) +
  geom_smooth(aes(linetype = data, fill = data, colour = data))
ggsave("plot/gpt2_webtext_xl_xl40k.freq_power.pdf", plot=p_gpt2_wb_xl_xl40k)

# Compare all data wthout k40
d_gpt2_wb_sm_me_la_xl <- rbindlist(list(d_gpt2_webtext,
                                       d_gpt2_small, d_gpt2_medium, d_gpt2_large, d_gpt2_xl))
p_gpt2_wb_sm_me_la_xl <- ggplot(d_gpt2_wb_sm_me_la_xl, aes(freq, power)) +
  geom_smooth(aes(linetype = data, fill = data, colour = data))
ggsave("plot/gpt2_webtext_small_medium_large_xl.freq_power.pdf", plot=p_gpt2_wb_sm_me_la_xl)

# Compare all data with k40
d_gpt2_wb_sm_me_la_xl_k40 <- rbindlist(list(d_gpt2_webtext,
                                        d_gpt2_small40k, d_gpt2_medium40k, d_gpt2_large40k, d_gpt2_xl40k))
p_gpt2_wb_sm_me_la_xl_k40 <- ggplot(d_gpt2_wb_sm_me_la_xl_k40, aes(freq, power)) +
  geom_smooth(aes(linetype = data, fill = data, colour = data))
ggsave("plot/gpt2_webtext_small_medium_large_xl.k40.freq_power.pdf", plot=p_gpt2_wb_sm_me_la_xl_k40)