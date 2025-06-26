import asyncio
import random

async def producer(queue, name):
    # """"""生产者：向队列中放入数据""""""
    for i in range(5):
        await asyncio.sleep(random.uniform(0.5, 1.5))  # 模拟生产时间
        item = f"{name}-产品{i}"
        await queue.put(item)
        print(f"生产者 {name} 生产了: {item}")

async def consumer(queue, name):
    # """"""消费者：从队列中取出数据""""""
    while True:
        try:
            # 等待1秒，如果没有数据就超时
            item = await asyncio.wait_for(queue.get(), timeout=1.0)
            await asyncio.sleep(random.uniform(0.3, 0.8))  # 模拟处理时间
            print(f"消费者 {name} 处理了: {item}")
            queue.task_done()
        except asyncio.TimeoutError:
            print(f"消费者 {name} 超时退出")
            break

async def main():
    # 创建队列
    queue = asyncio.Queue(maxsize=3)

    # 创建生产者和消费者
    producers = [
        producer(queue,"生产商A"),
        producer(queue,"生产商B"),
    ]

    consumers = [
        consumer(queue, "工人1"),
        consumer(queue, "工人2"),
        consumer(queue, "工人3"),
    ]

    # 并发运行
    await asyncio.gather(*producers, *consumers)

asyncio.run(main())