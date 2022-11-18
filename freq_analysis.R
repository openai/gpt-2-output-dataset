require(ggplot2)
require(data.table)

d_webtext <- fread("plot/webtext_freq_power_1k.csv")
d_webtext$source <- "webtext"
p <- ggplot(d, aes(freq, power)) + geom_smooth()
ggsave("plot/webtext_freq_power_1k_smooth.pdf", plot=p)

# small-117M
d_small <- fread("plot/small-117M_freq_power_1k.csv")
d_small$source <- "small-117M"
d_small_k40 <- fread("plot/small-117M-k40_freq_power_1k.csv")
d_small_k40$source <- "small-117M-k40"

p_small <- ggplot(d_small, aes(freq, power)) + geom_smooth()
ggsave("plot/small-117M_freq_power_1k_smooth.pdf", plot=p_small)

# Medium-345M
d_medium <- fread("plot/medium-345M_freq_power_1k.csv")
d_medium$source <- "medium-345M"
d_medium_k40 <- fread("plot/medium-345M-k40_freq_power_1k.csv")
d_medium_k40$source <- "medium-345M-k40"

# Large-762M
d_large <- fread("plot/large-762M_freq_power_1k.csv")
d_large$source <- "large-762M"
d_large_k40 <- fread("plot/large-762M-k40_freq_power_1k.csv")
d_large_k40$source <- "large-762M-k40"

# xl-1542M
d_xl <- fread("plot/xl-1542M_freq_power_1k.csv")
d_xl$source <- "xl-1542M"
d_xl_k40 <- fread("plot/xl-1542M-k40_freq_power_1k.csv")
d_xl_k40$source <- "xl-1542M-k40"


# All combined
dc <- rbindlist(list(d_webtext, d_small, d_medium, d_large, d_xl))
pc <- ggplot(dc, aes(freq, power)) + geom_smooth(aes(linetype = source, fill = source, colour = source))
ggsave("plot/webtext_all_freq_power_1k_smooth.pdf", plot=pc)

dc_k40 <- rbindlist(list(d_webtext, d_small_k40, d_medium_k40, d_large_k40, d_xl_k40))
pc_k40 <- ggplot(dc_k40, aes(freq, power)) + geom_smooth(aes(linetype = source, fill = source, colour = source))
ggsave("plot/webtext_all-k40_freq_power_1k_smooth.pdf", plot=pc_k40)


# Small and small-k40
d_small_c <- rbindlist(list(d_webtext, d_small, d_small_k40))
p_small_c <- ggplot(d_small_c, aes(freq, power)) + geom_smooth(aes(linetype = source, fill = source, colour = source))
ggsave("plot/webtext_small_small-k40_freq_power_1k_smooth.pdf", plot=p_small_c)

d_medium_c <- rbindlist(list(d_webtext, d_medium, d_medium_k40))
p_medium_c <- ggplot(d_medium_c, aes(freq, power)) + geom_smooth(aes(linetype = source, fill = source, colour = source))
ggsave("plot/webtext_medium_medium-k40_freq_power_1k_smooth.pdf", plot=p_medium_c)

d_large_c <- rbindlist(list(d_webtext, d_large, d_large_k40))
p_large_c <- ggplot(d_large_c, aes(freq, power)) + geom_smooth(aes(linetype = source, fill = source, colour = source))
ggsave("plot/webtext_large_large-k40_freq_power_1k_smooth.pdf", plot=p_large_c)

d_xl_c <- rbindlist(list(d_webtext, d_xl, d_xl_k40))
p_xl_c <- ggplot(d_xl_c, aes(freq, power)) + geom_smooth(aes(linetype = source, fill = source, colour = source))
ggsave("plot/webtext_xl_xl-k40_freq_power_1k_smooth.pdf", plot=p_xl_c)