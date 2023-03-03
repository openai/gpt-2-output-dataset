require("data.table")
require("ggplot2")
require("stringr")

require("fpp")
require("forecast")


## Old way
# Read entropy data
# require("progress") # tqdm for R
# file_name <- "data/small-117M.test.model=gpt2.nll"
# file_conn <- file(file_name, "r")
# lines <- readLines(file_conn)
# pb <- progress_bar$new(format = "  downloading [:bar] :percent eta: :eta",
#                        total = 100)
# pb$tick(0)
# data <- data.table()
# for (i in 1:length(lines)) {
#   entropy = as.numeric(str_split(lines[i], " ")[[1]])
#   seriesID = rep(i, length(entropy))
#   data <- rbindlist(list(data, data.table(seriesID = seriesID, entropy = entropy)))
#   pb$tick()
# }
## 不好用，3/2/2023


# Read new stat.csv data
dt <- fread("stat_test/small-117M.test.model=gpt2.nll.stat.csv")
dt.test <- dt[, {
                 res1 = Box.test(entropy)
                 res2 = adf.test(entropy)
                 res3 = kpss.test(entropy)
                 res4 = pp.test(entropy)
                 .(boxpval = res1$p.value, adfpval = res2$p.value, kpsspval = res3$p.value, pppval = res4$p.value)
               }, by = .(series_id)]


entropy <- dt[series_id==0]$entropy
boxpval <- Box.test(entropy)
adfpval <- adf.test(entropy)
kpsspval <- kpss.test(entropy)
pppval <- pp.test(entropy)