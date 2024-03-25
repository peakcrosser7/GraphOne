/*
 * This file is open source software, licensed to you under the terms
 * of the Apache License, Version 2.0 (the "License").  See the NOTICE file
 * distributed with this work for additional information regarding copyright
 * ownership.  You may not use this file except in compliance with the License.
 *
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/*
 * Copyright 2016 ScyllaDB
 */

#pragma once

#include <iosfwd>
#include <array>
#include <mutex>
#include <cstring>
#include <vector>
#include <algorithm>

#include <link.h>
#include <execinfo.h>
#include <unistd.h>
#include <signal.h>

namespace graph_one {

struct shared_object {
    std::string name;
    uintptr_t begin;
    uintptr_t end;  // C++-style, last addr + 1
};

/// @brief 栈帧
struct frame {
    const shared_object* so;
    uintptr_t addr;
    std::string symbols;
};

bool operator==(const frame& a, const frame& b);


class saved_backtrace {
   public:
    using vector_type = std::array<frame, 64>;

   private:
    vector_type _frames;

   public:
    saved_backtrace() = default;
    saved_backtrace(vector_type f) : _frames(std::move(f)) {}
    size_t hash() const;

    friend std::ostream& operator<<(std::ostream& out, const saved_backtrace&);

    bool operator==(const saved_backtrace& o) const {
        return _frames == o._frames;
    }

    bool operator!=(const saved_backtrace& o) const { return !(*this == o); }
};

saved_backtrace current_backtrace() noexcept;
std::ostream& operator<<(std::ostream& out, const saved_backtrace& b);

}  // namespace graph_one

namespace std {

template <>
struct hash<graph_one::saved_backtrace> {
    size_t operator()(const graph_one::saved_backtrace& b) const {
        return b.hash();
    }
};

}  // namespace std

namespace graph_one {
//
// Collection of async-signal safe printing functions.
//

// Outputs string to stderr.
// Async-signal safe.
inline void print_safe(const char* str, size_t len) noexcept {
    while (len) {
        auto result = write(STDERR_FILENO, str, len);
        if (result > 0) {
            len -= result;
            str += result;
        } else if (result == 0) {
            break;
        } else {
            if (errno == EINTR) {
                // retry
            } else {
                break;  // what can we do?
            }
        }
    }
}

// Outputs string to stderr.
// Async-signal safe.
inline void print_safe(const char* str) noexcept {
    print_safe(str, strlen(str));
}

// Fills a buffer with a zero-padded hexadecimal representation of an integer.
// For example, convert_zero_padded_hex_safe(buf, 4, uint16_t(12)) fills the
// buffer with "000c".
/// @brief 将数字转换为16进制字符串
/// @tparam Integral 数字类型
/// @param[out] buf 输出的缓存
/// @param bufsz 缓存大小
/// @param n 待转换数字
template <typename Integral>
void convert_zero_padded_hex_safe(char* buf, size_t bufsz,
                                  Integral n) noexcept {
    const char* digits = "0123456789abcdef";
    memset(buf, '0', bufsz);
    unsigned i = bufsz;
    while (n) {
        buf[--i] = digits[n & 0xf];
        n >>= 4;
    }
}

// Prints zero-padded hexadecimal representation of an integer to stderr.
// For example, print_zero_padded_hex_safe(uint16_t(12)) prints "000c".
// Async-signal safe.
template <typename Integral>
void print_zero_padded_hex_safe(Integral n) noexcept {
    static_assert(
        std::is_integral<Integral>::value && !std::is_signed<Integral>::value,
        "Requires unsigned integrals");

    char buf[sizeof(n) * 2];
    convert_zero_padded_hex_safe(buf, sizeof(buf), n);
    print_safe(buf, sizeof(buf));
}

// Fills a buffer with a decimal representation of an integer.
// The argument bufsz is the maximum size of the buffer.
// For example, print_decimal_safe(buf, 16, 12) prints "12".
template <typename Integral>
size_t convert_decimal_safe(char* buf, size_t bufsz, Integral n) noexcept {
    static_assert(
        std::is_integral<Integral>::value && !std::is_signed<Integral>::value,
        "Requires unsigned integrals");

    char tmp[sizeof(n) * 3];
    unsigned i = bufsz;
    do {
        tmp[--i] = '0' + n % 10;
        n /= 10;
    } while (n);
    memcpy(buf, tmp + i, sizeof(tmp) - i);
    return sizeof(tmp) - i;
}

// Prints decimal representation of an integer to stderr.
// For example, print_decimal_safe(12) prints "12".
// Async-signal safe.
template <typename Integral>
void print_decimal_safe(Integral n) noexcept {
    char buf[sizeof(n) * 3];
    unsigned i = sizeof(buf);
    auto len = convert_decimal_safe(buf, i, n);
    print_safe(buf, len);
}

inline int dl_iterate_phdr_callback(struct dl_phdr_info* info,
                                    size_t /* size */, void* data) {
    std::size_t total_size{0};
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const auto hdr = info->dlpi_phdr[i];

        // Only account loadable, executable (text) segments
        if (hdr.p_type == PT_LOAD && (hdr.p_flags & PF_X) == PF_X) {
            total_size += hdr.p_memsz;
        }
    }

    reinterpret_cast<std::vector<shared_object>*>(data)->push_back(
        {info->dlpi_name, info->dlpi_addr, info->dlpi_addr + total_size});

    return 0;
}

inline std::vector<shared_object> enumerate_shared_objects() {
    std::vector<shared_object> shared_objs;
    dl_iterate_phdr(dl_iterate_phdr_callback, &shared_objs);

    return shared_objs;
}

// Accumulates an in-memory backtrace and flush to stderr eventually.
// Async-signal safe.
/// @brief 程序回溯信息缓存
class backtrace_buffer {
    static constexpr unsigned _max_size = 8 << 10;
    unsigned _pos = 0;
    char _buf[_max_size];

    static const std::vector<shared_object> shared_objects;
    static const shared_object uknown_shared_object;  

    // If addr doesn't seem to belong to any of the provided shared objects, it
    // will be considered as part of the executable.
    frame decorate(uintptr_t addr) {
        char** s = backtrace_symbols((void**)&addr, 1);
        std::string symbol(*s);
        free(s);

        // If the shared-objects are not enumerated yet, or the enumeration
        // failed return the addr as-is with a dummy shared-object.
        if (shared_objects.empty()) {
            return {&uknown_shared_object, addr, std::move(symbol)};
        }

        auto it = std::find_if(shared_objects.begin(), shared_objects.end(),
                            [&](const shared_object& so) {
                                return addr >= so.begin && addr < so.end;
                            });

        // Unidentified addresses are assumed to originate from the executable.
        auto& so = it == shared_objects.end() ? shared_objects.front() : *it;
        return {&so, addr - so.begin, std::move(symbol)};
    }

    // Invokes func for each frame passing it as argument.
    template <typename Func>
    void backtrace(Func&& func) noexcept(noexcept(func(frame()))) {
        constexpr size_t max_backtrace = 100;
        void* buffer[max_backtrace];
        int n = ::backtrace(buffer, max_backtrace);
        for (int i = 0; i < n; ++i) {
            auto ip = reinterpret_cast<uintptr_t>(buffer[i]);
            func(decorate(ip - 1));
        }
    }

   public:
    /// @brief 输出回溯信息到stderr
    void flush() noexcept {
        print_safe(_buf, _pos);
        _pos = 0;
    }

    /// @brief 添加字符串信息
    /// @param str 字符串
    /// @param len 字符串长度
    void append(const char* str, size_t len) noexcept {
        /// 超出缓存大小则输出
        if (_pos + len >= _max_size) {
            flush();
        }
        memcpy(_buf + _pos, str, len);
        _pos += len;
    }

    /// @brief 添加字符串信息
    /// @param str 字符串
    void append(const char* str) noexcept { append(str, strlen(str)); }

    template <typename Integral>
    void append_decimal(Integral n) noexcept {
        char buf[sizeof(n) * 3];
        auto len = convert_decimal_safe(buf, sizeof(buf), n);
        append(buf, len);
    }

    /// @brief 添加16进制地址信息
    /// @tparam Integral 地址类型
    /// @param ptr 地址
    template <typename Integral>
    void append_hex(Integral ptr) noexcept {
        char buf[sizeof(ptr) * 2];
        // 转换为16进制字符串
        convert_zero_padded_hex_safe(buf, sizeof(buf), ptr);
        append(buf, sizeof(buf));
    }

    /// @brief 添加回溯信息
    void append_backtrace() noexcept {
        backtrace([this](frame f) {
            append("  ");
            if (!f.so->name.empty()) {
                append(f.so->name.c_str(), f.so->name.size());
                append("+");
            }

            append("0x");
            // 添加栈帧地址
            append_hex(f.addr);
            append("\n");
        });
    }
};

inline const std::vector<shared_object> backtrace_buffer::shared_objects =
        enumerate_shared_objects();
inline const shared_object backtrace_buffer::uknown_shared_object = {
        "", 0, std::numeric_limits<uintptr_t>::max()};


// Installs handler for Signal which ensures that Func is invoked only once
// in the whole program and that after it is invoked the default handler is
// restored.
/// @brief 安装一次性信号处理程序
/// @tparam Signal 信号编号
/// @tparam Func 处理函数
template <int Signal, void (*Func)()>
void init_backtrace() {
    // 用于确保信号处理函数Func只被调用1次
    static bool handled = false;
    static std::mutex lock;

    struct sigaction sa;
    // 定义信号处理程序
    sa.sa_sigaction = [](int sig, siginfo_t* /* info */, void* /* p */) {
        // 作用域锁
        std::lock_guard<std::mutex> g(lock);
        if (!handled) {
            handled = true;
            // 信号处理程序
            Func();
            // 设置信号为默认处理程序
            signal(sig, SIG_DFL);
        }
    };
    // 设置为所有信号掩码,以确保在号处理程序执行期间不会被其他信号中断
    sigfillset(&sa.sa_mask);
    // 使用sa_sigaction作为信号处理程序，并在系统调用被信号中断时自动重启系统调用
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    if (Signal == SIGSEGV) {
        // 在另一个堆栈上执行信号处理程序
        sa.sa_flags |= SA_ONSTACK;
    }
    // 将信号处理程序安装到指定的信号上
    auto r = ::sigaction(Signal, &sa, nullptr);
    if (r == -1) {
        throw std::system_error();
    }
}

/// @brief 输出回溯信息
/// @param buf 回溯信息缓存
inline void print_with_backtrace(backtrace_buffer& buf) noexcept {
    buf.append(".\nBacktrace:\n");
    buf.append_backtrace();
    buf.flush();
}

/// @brief 输出回溯信息
/// @param cause 回溯造成的原因
inline void print_with_backtrace(const char* cause) noexcept {
    backtrace_buffer buf;
    // 添加原因
    buf.append(cause);
    print_with_backtrace(buf);
}

/// @brief 信号处理函数
inline void sigsegv_action() noexcept {
    print_with_backtrace("Segmentation fault");
}

inline void sigabrt_action() noexcept { 
    print_with_backtrace("Aborting"); 
}

inline void install_oneshot_signal_handlers() {
    // Mask most, to prevent threads (esp. dpdk helper threads)
    // from servicing a signal.  Individual reactors will unmask signals
    // as they become prepared to handle them.
    //
    // We leave some signals unmasked since we don't handle them ourself.

    // 信号集
    sigset_t sigs;
    // 初始化包含所有信号
    sigfillset(&sigs);
    // 将特定信号删除
    for (auto sig : {SIGHUP, SIGQUIT, SIGILL, SIGABRT, SIGFPE, SIGSEGV, SIGALRM,
                     SIGCONT, SIGSTOP, SIGTSTP, SIGTTIN, SIGTTOU}) {
        sigdelset(&sigs, sig);
    }
    // 设置信号掩码
    // 设置信号集中的信号被阻塞,这些信号不会被线程接收
    pthread_sigmask(SIG_BLOCK, &sigs, nullptr);

    // 添加信号SIGSEGV,SIGABRT的信号处理函数
    init_backtrace<SIGSEGV, sigsegv_action>();
    init_backtrace<SIGABRT, sigabrt_action>();
}

}  // namespace graph_one
