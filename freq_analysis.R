require(ggplot2)
require(data.table)

d_webtext <- fread("plot/webtext_freq_power_1k.csv")
setnames(d_webtext, c("freq", "power"))
d_webtext$source <- "webtext"
p <- ggplot(d, aes(freq, power)) + geom_smooth()
ggsave("plot/webtext_freq_power_1k_smooth.pdf", plot=p)

# small-117M
d_small <- fread("plot/small-117M_freq_power_1k.csv")
d_small$source <- "small-117M"
p_small <- ggplot(d_small, aes(freq, power)) + geom_smooth()
ggsave("plot/small-117M_freq_power_1k_smooth.pdf", plot=p_small)

d_small_k40 <- fread("plot/small-117M-k40_freq_power_1k.csv")
d_small_k40$source <- "small-117M-k40"

# Medium-345M
d_medium <- fread("plot/medium-345M_freq_power_1k.csv")
d_medium$source <- "medium-345M"


# All combined
dc <- rbindlist(list(d_webtext, d_small, d_medium, d_small_k40))
pc <- ggplot(dc, aes(freq, power)) + geom_smooth(aes(linetype = source, fill = source, colour = source))
ggsave("plot/webtext_small_medium_freq_power_1k_smooth.pdf", plot=pc)


# Small and small-k40
d_small_c <- rbindlist(list(d_small, d_small_k40))
p_small_c <- ggplot(d_small_c, aes(freq, power)) + geom_smooth(aes(linetype = source, fill = source, colour = source))
ggsave("plot/small_small-k40_freq_power_1k_smooth.pdf", plot=p_small_c)