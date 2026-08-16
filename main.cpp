#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <print>
#include <ranges>
#include <span>
#include <spanstream>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>


/**
 * @brief: Unique Location Temperature Readings Stats
 *
 * - Given a csv file of some 10 million records Location;Temperature
 *   find the statistics (min, max, mean) for each location
 *   10'000 Unique Locations in the Dataset
 * - Temperature Readings are single decimal [-99.9, 99.9]
 * - Handle mean only
 *
 */

// streaming the File
// fstream object internal buffer (could be 4Kb - 8Kb)
// read line and store
// location to Stats Hashed Value

namespace
{
    struct Stats
    {
        std::size_t   count_{};
        float         min_{std::numeric_limits<float>::max()};
        float         max_{0.0};
        float         sum_{};
    };

    struct string_hasher
    {
        using is_transparent = void;

        [[nodiscard]] size_t operator()(std::string_view text) const noexcept {
            return std::hash<std::string_view>{}(text);
        }

    };

    using StringRecords = std::unordered_map<std::string, Stats>;
    using StringRecordsHeteroLookup = std::unordered_map<std::string, Stats, string_hasher, std::equal_to<>>;


    void process_base1()
    {
        constexpr auto filepath = "./sample_files/measurements_10_mill.txt";
        errno = 0;
        std::ifstream stream(filepath, std::ifstream::in);

        if (!stream.is_open()) {
            throw std::system_error(errno, std::generic_category(), filepath);
        }

        StringRecords       records_table;
        std::string         loc_name, temp_valStr;

        while (std::getline(stream, loc_name, ';') &&
                std::getline(stream, temp_valStr, '\n'))
        {

            float const temp_val = std::stof(temp_valStr);

            auto [it_stat, is_inserted] = records_table.try_emplace(
                loc_name.data(),
                Stats {
                    .count_ = 1,
                    .min_ = temp_val,
                    .max_ = temp_val,
                    .sum_ = temp_val
                }
            );

            if (!is_inserted)
            {
                auto& [count_val, min_val, max_val, sum_val] = it_stat->second;
                sum_val += temp_val;
                min_val = std::min(min_val, temp_val);
                max_val = std::max(max_val, temp_val);
                ++count_val;
            }

        }

        // std::println("\nRecords in Record Table: {}", records_table.size());

        stream.close();
    }

    void process_base2()
    {
        errno = 0;
        constexpr auto filepath = "./sample_files/measurements_10_mill.txt";
        std::ifstream stream(filepath, std::ifstream::in);

        if (!stream.is_open()) {
            throw std::system_error(errno, std::generic_category(), filepath);
        }

        StringRecordsHeteroLookup       loc_statTable;
        std::string                     line;
        std::string_view                loc_name, temp_valStr;

        while (std::getline(stream, line))
        {
            auto const it_delim = std::ranges::find(line, ';');
            loc_name = {line.begin(), it_delim};
            temp_valStr = {it_delim + 1, line.end()};


            float const temp_val = std::stof(temp_valStr.data());

            if (auto it_stat = loc_statTable.find(loc_name);
                it_stat == loc_statTable.end()) {
                loc_statTable.emplace(loc_name,
                    Stats {
                        .count_ = 1,
                        .min_ = temp_val,
                        .max_ = temp_val,
                        .sum_ = temp_val
                    }
                );
            }
            else {
                auto& [count_val, min_val, max_val, sum_val] = it_stat->second;
                sum_val += temp_val;
                min_val = std::min(min_val, temp_val);
                max_val = std::max(max_val, temp_val);
                ++count_val;
            }
        }

        stream.close();
    }

    // Approach 3: Memory Mapping of the CSV File
    // 1. virtually map the file into our process' virtual address space
    //    - to avoid overhead from invoking syscall to read data
    //    - avoids copying file memory into kernel buffer and then into our userspace
    //      ifstream object buffer
    // 2. non-owning view over the bytes we process (span stream for raw bytes)


    /**
    *   void *mmap(size_t length;
    *              void addr[length], size_t length, int prot, int flags,
    *              int fd, off_t offset);
    *      - file length needed
    *      - fd   needed
    *
    *   FileResource - RAII Wrapper for the file resource that is opened
    *   MappedEntry  - Encapsulates Memory Mapping Behavior
    *   - open file
    *   - memory map
    *   - unmap map
    *
    *
    *   int munmap(size_t length;
    *              void addr[length], size_t length);
     */

    // Ensure kernel definitions exist
    #ifndef MAP_HUGETLB
        #define MAP_HUGETLB 0x40000
    #endif

    #ifndef MAP_HUGE_SHIFT
        #define MAP_HUGE_SHIFT 26
    #endif

    constexpr int mmap_huge_2mb = (21 << MAP_HUGE_SHIFT);
    constexpr int mmap_huge_1gb = (30 << MAP_HUGE_SHIFT);


    class FileResource
    {
        struct MMappedEntry
        {
            void* mem_ptr_{nullptr};
            std::size_t size_{0};

            MMappedEntry() = default;

            ~MMappedEntry() noexcept {
                cleanup();
            }

            void cleanup() noexcept
            {
                if (mem_ptr_ && mem_ptr_ != MAP_FAILED) {
                    ::munmap(mem_ptr_, size_);
                }
                mem_ptr_ = nullptr;
                size_ = 0;
            }

            void mmap_file(int const fd, std::size_t const file_size, int const permissions, int const flags)
            {
                cleanup();

                mem_ptr_ = ::mmap(nullptr, file_size, permissions, flags, fd, 0);

                if (mem_ptr_ == MAP_FAILED) {
                    throw std::system_error(errno, std::system_category(), "mmap returned MAP_FAILED");
                }
                size_ = file_size;
            }

            [[nodiscard]] std::span<char> get_span() const noexcept
            {
                if (mem_ptr_ == MAP_FAILED || mem_ptr_ == nullptr) return {};
                return std::span<char>{static_cast<char*>(mem_ptr_), size_};
            }
        };

    private:
        int fd_{-1};
        std::filesystem::path path_{};
        std::size_t file_size_{0};
        MMappedEntry mapped_entry_;

    public:
        explicit FileResource(std::string_view const path) : path_(path)
        {
            fd_ = ::open(path_.c_str(), O_RDONLY);
            if (fd_ == -1) {
                throw std::system_error(errno, std::system_category(), "file could not be opened");
            }

            file_size_ = std::filesystem::file_size(path_);
            if (file_size_ <= 0) {
                throw std::system_error(errno, std::system_category(), "file is empty or invalid");
            }
        }

        void memory_mapFile(int const permissions, int const flags) {
            mapped_entry_.mmap_file(fd_, file_size_, permissions, flags);
        }

        [[nodiscard]] std::span<char const> get_bufferSpan() const noexcept {
            return mapped_entry_.get_span();
        }

        ~FileResource() noexcept {
            if (fd_ != -1) {
                ::close(fd_);
            }
        }

        FileResource(const FileResource&) = delete;
        FileResource& operator=(const FileResource&) = delete;
    };

    void process_baseMMAP()
    {
        constexpr auto filepath = "./sample_files/measurements_10_mill.txt";

        FileResource file_handle(filepath);
        file_handle.memory_mapFile(PROT_READ, MAP_PRIVATE);

        auto mmap_span = file_handle.get_bufferSpan();
        std::ispanstream stream{mmap_span};

        StringRecordsHeteroLookup records_table;

        StringRecordsHeteroLookup       loc_statTable;
        std::string                     line;
        std::string_view                loc_name, temp_valStr;

        while (std::getline(stream, line))
        {
            auto const it_delim = std::ranges::find(line, ';');
            loc_name = {line.begin(), it_delim};
            temp_valStr = {it_delim + 1, line.end()};


            float const temp_val = std::stof(temp_valStr.data());

            if (auto it_stat = loc_statTable.find(loc_name);
                it_stat == loc_statTable.end()) {
                loc_statTable.emplace(loc_name,
                    Stats {
                        .count_ = 1,
                        .min_ = temp_val,
                        .max_ = temp_val,
                        .sum_ = temp_val
                    }
                );
                }
            else {
                auto& [count_val, min_val, max_val, sum_val] = it_stat->second;
                sum_val += temp_val;
                min_val = std::min(min_val, temp_val);
                max_val = std::max(max_val, temp_val);
                ++count_val;
            }
        }

    }

    void process_mmap()
    {
        constexpr auto filepath = "./sample_files/measurements_10_mill.txt";

        FileResource file_handle(filepath);
        file_handle.memory_mapFile(PROT_READ, MAP_PRIVATE);

        auto mmap_span = file_handle.get_bufferSpan();

        StringRecordsHeteroLookup records_table;

        auto it_curr = mmap_span.begin();

        while (it_curr != mmap_span.end())
        {
            auto const it_recordEnd = std::ranges::find(it_curr, std::unreachable_sentinel, '\n');
            auto const it_fieldDelim = std::ranges::find(it_curr, it_recordEnd, ';');

            std::string_view loc_name = {it_curr, it_fieldDelim};
            std::string_view temperature_str = {it_fieldDelim + 1, it_recordEnd};

            auto const temperature_val = std::stof(std::string(temperature_str));

            if (auto it_record = records_table.find(loc_name);
                it_record == records_table.end())
            {
                records_table.emplace(loc_name,
                    Stats {
                    .count_ = 1,
                    .min_ = temperature_val,
                    .max_ = temperature_val,
                    .sum_ = temperature_val
                });
            }
            else
            {
                auto& [count, min, max, sum] = it_record->second;
                ++count;
                min = std::min(min, temperature_val);
                max = std::max(max, temperature_val);
                sum += temperature_val;
            }

            it_curr = it_recordEnd + 1;
        }

        // std::println("\nRecords in Record Table: {}", records_table.size());

    }

    void process_mmapFromChar()
    {
        constexpr auto filepath = "./sample_files/measurements_10_mill.txt";

        FileResource file_handle(filepath);
        file_handle.memory_mapFile(PROT_READ, MAP_PRIVATE);

        auto mmap_span = file_handle.get_bufferSpan();
        std::ispanstream mmap_stream(mmap_span);

        StringRecordsHeteroLookup records_table;

        auto it_curr = mmap_span.begin();
        float temperature_val{};

        while (it_curr != mmap_span.end())
        {
            auto const it_recordEnd = std::ranges::find(it_curr, std::unreachable_sentinel, '\n');
            auto const it_fieldDelim = std::ranges::find(it_curr, it_recordEnd, ';');

            std::string_view loc_name = {it_curr, it_fieldDelim};
            std::string_view temperature_str = {it_fieldDelim + 1, it_recordEnd};

            auto [ptr, ec] = std::from_chars(
                temperature_str.begin(),
                temperature_str.end(),
                temperature_val);

            if (ec != std::errc{}) {
                throw std::invalid_argument("Invalid string provided for float conversion to std::from_chars");
            }


            if (auto it_record = records_table.find(loc_name);
                it_record == records_table.end())
            {
                records_table.emplace(loc_name,
                    Stats {
                    .count_ = 1,
                    .min_ = temperature_val,
                    .max_ = temperature_val,
                    .sum_ = temperature_val
                });
            }
            else
            {
                auto& [count, min, max, sum] = it_record->second;
                ++count;
                min = std::min(min, temperature_val);
                max = std::max(max, temperature_val);
                sum += temperature_val;
            }

            it_curr = it_recordEnd + 1;
        }

        // std::println("\nRecords in Record Table: {}", records_table.size());

    }

    void process_mmapMemChr()
    {
        constexpr auto filepath = "./sample_files/measurements_10_mill.txt";

        FileResource file_handle(filepath);
        file_handle.memory_mapFile(PROT_READ, MAP_PRIVATE);

        auto mmap_span = file_handle.get_bufferSpan();
        StringRecordsHeteroLookup records_table;

        auto curr_ptr = mmap_span.data();
        auto const end_ptr = mmap_span.data() + mmap_span.size();
        float temperature_val{};

        while (curr_ptr < end_ptr)
        {
            auto const remaining_bytes = static_cast<std::size_t>(end_ptr - curr_ptr);
            auto const field_delim = static_cast<char const*>(std::memchr(curr_ptr, ';', remaining_bytes));

            if (!field_delim) break;

            std::string_view loc_name = { curr_ptr, field_delim };

            auto const post_delim_ptr = field_delim + 1;
            auto const post_delim_bytes = static_cast<std::size_t>(end_ptr - post_delim_ptr);
            auto const record_end = static_cast<char const*>(std::memchr(post_delim_ptr, '\n', post_delim_bytes));

            char const* actual_end = record_end ? record_end : end_ptr;
            std::string_view temperature_str = { post_delim_ptr, actual_end };

            auto [ptr, ec] = std::from_chars(
                temperature_str.data(),
                temperature_str.data() + temperature_str.size(),
                temperature_val);

            if (ec != std::errc{}) {
                throw std::invalid_argument("Invalid string provided for float conversion to std::from_chars");
            }

            if (auto it_record = records_table.find(loc_name);
                it_record == records_table.end())
            {
                records_table.emplace(loc_name,
                    Stats {
                    .count_ = 1,
                    .min_ = temperature_val,
                    .max_ = temperature_val,
                    .sum_ = temperature_val
                });
            }
            else
            {
                auto& [count, min, max, sum] = it_record->second;
                ++count;
                min = std::min(min, temperature_val);
                max = std::max(max, temperature_val);
                sum += temperature_val;
            }

            curr_ptr = actual_end;
            if (curr_ptr < end_ptr && *curr_ptr == '\n') {
                curr_ptr++;
            }
        }

        // std::println("\nRecords in Record Table: {}", records_table.size());
    }


    void process_mmapCharReserved()
    {
        constexpr auto filepath = "./sample_files/measurements_10_mill.txt";

        FileResource file_handle(filepath);
        file_handle.memory_mapFile(PROT_READ, MAP_PRIVATE);

        auto mmap_span = file_handle.get_bufferSpan();
        std::ispanstream mmap_stream(mmap_span);

        StringRecordsHeteroLookup records_table;
        records_table.reserve(557);

        auto it_curr = mmap_span.begin();
        float temperature_val{};

        while (it_curr != mmap_span.end())
        {
            auto const it_recordEnd = std::ranges::find(it_curr, std::unreachable_sentinel, '\n');
            auto const it_fieldDelim = std::ranges::find(it_curr, it_recordEnd, ';');

            std::string_view loc_name = {it_curr, it_fieldDelim};
            std::string_view temperature_str = {it_fieldDelim + 1, it_recordEnd};

            auto [ptr, ec] = std::from_chars(
                temperature_str.begin(),
                temperature_str.end(),
                temperature_val);

            if (ec != std::errc{}) {
                throw std::invalid_argument("Invalid string provided for float conversion to std::from_chars");
            }


            if (auto it_record = records_table.find(loc_name);
                it_record == records_table.end())
            {
                records_table.emplace(loc_name,
                    Stats {
                    .count_ = 1,
                    .min_ = temperature_val,
                    .max_ = temperature_val,
                    .sum_ = temperature_val
                });
            }
            else
            {
                auto& [count, min, max, sum] = it_record->second;
                ++count;
                min = std::min(min, temperature_val);
                max = std::max(max, temperature_val);
                sum += temperature_val;
            }

            it_curr = it_recordEnd + 1;
        }

        // std::println("\nRecords in Record Table: {}", records_table.size());

    }


    struct StatsNominal
    {
        std::size_t   count_{};
        std::int16_t  min_{};
        std::int16_t  max_{};
        std::int32_t sum_{};
    };


    using RecordsNominal = std::unordered_map<std::string, StatsNominal, string_hasher, std::equal_to<>>;

    void process_mmapChResInt1()
    {
        constexpr auto filepath = "./sample_files/measurements_10_mill.txt";

        FileResource file_handle(filepath);
        file_handle.memory_mapFile(PROT_READ, MAP_PRIVATE);

        auto mmap_span = file_handle.get_bufferSpan();
        std::ispanstream mmap_stream(mmap_span);

        RecordsNominal records_table;
        records_table.reserve(557);

        auto it_curr = mmap_span.begin();
        std::int16_t temperature_val{};

        while (it_curr != mmap_span.end())
        {
            auto const it_recordEnd = std::ranges::find(it_curr, std::unreachable_sentinel, '\n');
            auto const it_fieldDelim = std::ranges::find(it_curr, it_recordEnd, ';');

            std::string_view loc_name = {it_curr, it_fieldDelim};
            std::string_view temperature_str = {it_fieldDelim + 1, it_recordEnd};

            auto const it_decimal = std::ranges::find(temperature_str, '.');

            if (it_decimal != std::ranges::end(temperature_str))
            {
                auto [it, ec] = std::from_chars(temperature_str.begin(),
                    it_decimal, temperature_val, 10);

                if (ec != std::errc{}) {
                    throw std::runtime_error("from_chars integer throw");
                }

                temperature_val *= 10;
                std::int16_t decimal_val{};
                auto [it_dec, ec_dec] = std::from_chars(it_decimal + 1,
                    temperature_str.end(), decimal_val, 10);

                temperature_val += decimal_val;

                if (*temperature_str.begin() == '-') {
                    temperature_val *= -1;
                }
            }



            if (auto it_record = records_table.find(loc_name);
                it_record == records_table.end())
            {
                records_table.emplace(loc_name,
                    StatsNominal {
                    .count_ = 1,
                    .min_ = temperature_val,
                    .max_ = temperature_val,
                    .sum_ = static_cast<std::int32_t>(temperature_val)
                });
            }
            else
            {
                auto& [count, min, max, sum] = it_record->second;
                ++count;
                min = std::min(min, temperature_val);
                max = std::max(max, temperature_val);
                sum += temperature_val;
            }

            it_curr = it_recordEnd + 1;
        }

    }


    // void callwithTime(auto&& callable) {
    //     auto const start = std::chrono::steady_clock::now();
    //     callable();
    //     auto const end = std::chrono::steady_clock::now();
    //     std::println("Elapsed Time: {}ms",
    //         std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    // }
}

int main(int const argc, char const* argv[])
{

    // auto const start = std::chrono::steady_clock::now();
    // process_base1();
    // process_baseMMAP();
    // process_mmapFromChar();
    // process_mmapMemChr();

    // process_mmapCharReserved();
    process_mmapChResInt1();

    // auto const end = std::chrono::steady_clock::now();
    // std::println("Elapsed Time: {}ms",
    //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
}


