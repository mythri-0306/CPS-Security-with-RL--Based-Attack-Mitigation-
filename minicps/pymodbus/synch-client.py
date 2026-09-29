#!/usr/bin/env python

"""
synch-client.py

value is passed either as a ``str`` or as a ``bool``. In case of ``str`` the value is
converted to an ``int`` to be written in a holding register
"""

import argparse
try:
    from pymodbus.client import ModbusTcpClient as ModbusClient
except ImportError:
    from pymodbus.client.sync import ModbusTcpClient as ModbusClient

from sys import argv

def _check_resp_ok(resp):
    if hasattr(resp, 'isError'):
        return not resp.isError()
    if hasattr(resp, 'function_code'):
        return resp.function_code < 0x80
    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, dest='ip', help='request ip')
    parser.add_argument('-p', type=int, dest='port',
            default=502, help='port number')
    parser.add_argument('-u', type=int, dest='unit',
            default=0, help='slave unit number')
    parser.add_argument('-t', type=str, dest='type',
            choices=['DI', 'CO', 'IR', 'HR'],
            help='request type')
    parser.add_argument('-m', type=str, dest='mode',
            choices=['r', 'w'],
            help='mode: read or write')
    parser.add_argument('-o', dest='offset', type=int,
            help='0-based modbus addressing offset',
            choices=range(0, 2000),
            default=0)
    parser.add_argument('--count', dest='count',
            help='count for multiple read and write',
            type=int, choices=range(1, 2000),
            default=1)
    parser.add_argument('-r', dest='register',
            help='list of int values', type=int,
            choices=range(0, 65536), nargs='+',
            default=0)
    parser.add_argument('-c', dest='coil',
            help='list of 0 (False) or 1 (True) int values', type=int, nargs='+',
            default=0, choices=[0, 1])

    args = parser.parse_args()

    client = ModbusClient(args.ip, port=args.port)
    client.connect()

    if args.mode == 'w':
        if args.type == 'HR':
            if args.count == 1:
                hr_write = client.write_register(args.offset, args.register[0])
                assert(_check_resp_ok(hr_write))
            else:
                hrs_write = client.write_registers(args.offset, args.register)
                assert(_check_resp_ok(hrs_write))

        elif args.type == 'CO':
            if args.count == 1:
                co_val = True if args.coil[0] == 1 else False
                co_write = client.write_coil(args.offset, co_val)
                assert(_check_resp_ok(co_write))
            else:
                coils = [True if c == 1 else False for c in args.coil]
                cos_write = client.write_coils(args.offset, coils)
                assert(_check_resp_ok(cos_write))

    elif args.mode == 'r':
        if args.type == 'HR':
            hr_read = client.read_holding_registers(args.offset, count=args.count)
            assert(_check_resp_ok(hr_read))
            print(hr_read.registers[0:args.count])

        elif args.type == 'IR':
            ir_read = client.read_input_registers(args.offset, count=args.count)
            assert(_check_resp_ok(ir_read))
            print(ir_read.registers[0:args.count])

        elif args.type == 'DI':
            di_read = client.read_discrete_inputs(args.offset, count=args.count)
            assert(_check_resp_ok(di_read))
            print(di_read.bits)

        elif args.type == 'CO':
            co_read = client.read_coils(args.offset, count=args.count)
            assert(_check_resp_ok(co_read))
            print(co_read.bits)

    client.close()

