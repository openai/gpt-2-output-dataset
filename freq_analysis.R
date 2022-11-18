require(ggplot2)
require(data.table)

d <- fread("plot/webtext_freq_power_1k.csv")
setnames(d, c("freq", "power"))

p <- ggplot(d, aes(freq, power)) + geom_smooth()

ggsave("plot/webtext_freq_power_1k_smooth.pdf", plot=p)