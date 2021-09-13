#include "MdHandler.h"

#include "IService.h"

namespace md_handler {
MdHandler::MdHandler(IService & service)
    : m_service(service)
{
}
void MdHandler::handle_packet(const Packet & packet)
{
    if (!is_heartbeat(packet)) {
        enqueue(packet);
    }
    uint32_t expected_as_last = is_heartbeat(packet) ? packet.get_seq_num() : packet.get_seq_num() - 1;
    if (m_last_message >= expected_as_last) {
        update_last(expected_as_last + packet.get_msg_count());
    }
    else {
        std::lock_guard<std::mutex> guard(m_last_number_access);
        if (!wait_late(expected_as_last)) {
            m_service.resend_messages(m_last_message + 1, expected_as_last - m_last_message);
        }
    }
    update_last(expected_as_last + packet.get_msg_count());
}
void MdHandler::handle_resend(const Packet & packet)
{
    enqueue(packet);
    handle_queue();
}

bool MdHandler::PacketComparator::operator()(const Packet & lhs, const Packet & rhs)
{
    return lhs.get_seq_num() > rhs.get_seq_num();
}

bool MdHandler::is_heartbeat(const Packet & packet)
{
    return packet.get_msg_count() == 0;
}

void MdHandler::update_last(uint32_t end_packet)
{
    m_last_message = std::max(m_last_message, end_packet);
    m_late_pockets.notify_one();
    handle_queue();
}

bool MdHandler::wait_late(uint32_t seq_num)
{
    std::unique_lock<std::mutex> lock(m_waiting_mutex);
    auto found = [this, seq_num] { return m_last_message >= seq_num; };
    m_late_pockets.wait_for(lock, WAIT_TL, found);
    return found();
}

void MdHandler::enqueue(const Packet & packet)
{
    std::lock_guard<std::mutex> guard(m_queue_access);
    m_queue.push(packet);
}
void MdHandler::handle_queue()
{
    std::lock_guard<std::mutex> guard(m_queue_access);
    while (!m_queue.empty() && m_queue.top().get_seq_num() <= m_expected_number) {
        Packet least = m_queue.top();
        m_queue.pop();
        handle_messages(least.get_seq_num() + least.get_msg_count());
    }
}
void MdHandler::handle_messages(uint32_t end_packet)
{
    for (uint16_t n = m_expected_number; n < end_packet; ++n) {
        m_service.handle_message(n);
    }
    m_expected_number = std::max(m_expected_number, end_packet);
}
} // namespace md_handler
