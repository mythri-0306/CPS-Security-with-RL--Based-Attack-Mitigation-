#!/usr/bin/env python

import argparse
import sys

try:
    from pymodbus.server import StartTcpServer
except ImportError:
    try:
        import importlib
        StartTcpServer = importlib.import_module('pymodbus.server.async').StartTcpServer
    except Exception:
        from pymodbus.server.asynchronous import StartTcpServer

try:
    from pymodbus.datastore import ModbusSequentialDataBlock
    from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
    _USE_PYMODBUS_V2_OR_OLDER = True
except (ImportError, TypeError):
    _USE_PYMODBUS_V2_OR_OLDER = False

if not _USE_PYMODBUS_V2_OR_OLDER:
    try:
        from pymodbus.datastore import ModbusSequentialDataBlock
        from pymodbus.datastore import ModbusDeviceContext, ModbusServerContext
    except ImportError:
        pass

try:
    from pymodbus.device import ModbusDeviceIdentification
except ImportError:
    ModbusDeviceIdentification = None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, dest='ip', help='server ip')
    parser.add_argument('-p', type=int, dest='port',
            default=502, help='port number')
    parser.add_argument('-m', type=int, dest='mode', choices=[1],
            default=1, help='mode')
    parser.add_argument('-d', type=int, dest='discrete_inputs',
            choices=range(1, 1000),
            default=10,
            help='number of discrete inputs')
    parser.add_argument('-c', type=int, dest='coils',
            choices=range(1, 1000),
            default=10,
            help='number of coils')
    parser.add_argument('-r', type=int, dest='input_registers',
            choices=range(1, 1000),
            default=10,
            help='number of input registers')
    parser.add_argument('-R', type=int, dest='holding_registers',
            choices=range(1, 1000),
            default=10,
            help='number of holding registers')

    args = parser.parse_args()

    # Create datastore based on pymodbus version
    try:
        store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * args.discrete_inputs),
            co=ModbusSequentialDataBlock(0, [0] * args.coils),
            ir=ModbusSequentialDataBlock(0, [0] * args.input_registers),
            hr=ModbusSequentialDataBlock(0, [0] * args.holding_registers),
            zero_mode=False,
        )
        context = ModbusServerContext(slaves=store, single=True)
    except Exception:
        # pymodbus 3.x
        from pymodbus.datastore import ModbusDeviceContext, ModbusServerContext, ModbusSequentialDataBlock
        store = ModbusDeviceContext(
            di=ModbusSequentialDataBlock(1, [0] * max(args.discrete_inputs, 100)),
            co=ModbusSequentialDataBlock(1, [0] * max(args.coils, 100)),
            ir=ModbusSequentialDataBlock(1, [0] * max(args.input_registers, 100)),
            hr=ModbusSequentialDataBlock(1, [0] * max(args.holding_registers, 100)),
        )
        context = ModbusServerContext(devices=store)

    identity = None
    if ModbusDeviceIdentification is not None:
        identity = ModbusDeviceIdentification()
        identity.VendorName = 'Pymodbus'
        identity.ProductCode = 'PM'
        identity.VendorUrl = 'http://github.com/bashwork/pymodbus/'
        identity.ProductName = 'Pymodbus Server'
        identity.ModelName = 'Pymodbus Server'
        identity.MajorMinorRevision = '1.0'

    if args.mode == 1:
        if identity is not None:
            StartTcpServer(context=context, identity=identity,
                           address=(args.ip, args.port))
        else:
            StartTcpServer(context=context,
                           address=(args.ip, args.port))


