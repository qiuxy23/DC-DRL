import redis

pool = redis.ConnectionPool(host='localhost', port=6379, decode_responses=True)
r = redis.Redis(connection_pool=pool)

r.delete('hash')
r.delete('agent_list')


r.sadd('agent_list', 0)

r.hset('hash', 'g', '0')
r.hset('hash', 'f_0', '-1000000')

# print(r.hexists('hash', 'g'))

r.hset('hash', 'f_avg', '-1000000')
# # whether is writeable, for atomicity
# r.hset('hash', 'flag', '0')
# # update with the fastest
r.hset('hash', 'n_g', '0')

# r.hdel('hash', 'p_0_0')

# r.sadd('agent_list', 1)
r.hset('hash', 'algo', 'dc')
r.hset('hash', 'env', 'MyEnv-10')
r.hset('hash', 'env', 'testbed')
