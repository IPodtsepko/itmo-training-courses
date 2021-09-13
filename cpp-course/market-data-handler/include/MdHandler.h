#pragma once

#include "Packet.h"

#include <cmath>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace md_handler {

class IService;

class MdHandler
{
public:
    MdHandler(IService & service);
    void handle_packet(const Packet & packet);
    void handle_resend(const Packet & packet);

private:
    IService & m_service;

    uint32_t m_last_message = 0;
    uint32_t m_expected_number = 1;

    static constexpr auto WAIT_TL = std::chrono::milliseconds(200);

    std::condition_variable m_late_pockets;
    std::mutex m_waiting_mutex;

    std::mutex m_queue_access;
    std::mutex m_last_number_access;

    struct PacketComparator
    {
        bool operator()(Packet const & lhs, Packet const & rhs);
    };
    std::priority_queue<Packet, std::vector<Packet>, PacketComparator> m_queue;

    static bool is_heartbeat(const Packet & packet);

    void update_last(uint32_t);

    bool wait_late(uint32_t seq_num);

    void enqueue(const Packet & packet);
    void handle_queue();
    void handle_messages(uint32_t);
};

} // namespace md_handler