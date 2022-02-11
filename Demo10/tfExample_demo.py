import struct
import tensorflow as tf


def read_text_file(text_file):
    lines = []
    with open(text_file, "r") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def text_to_binary(in_file, out_file):
    inputs = read_text_file(in_file)

    with open(out_file, 'wb') as writer:
        data_id = tf.train.Int64List(value=[int(inputs[0])])
        data = tf.train.BytesList(value=[bytes(' '.join(inputs[1:]), encoding='utf-8')])

        feature_dict = {
            "data_id": tf.train.Feature(int64_list=data_id),
            "data": tf.train.Feature(bytes_list=data)
        }
        features = tf.train.Features(feature=feature_dict)

        example = tf.train.Example(features=features)
        example_str = example.SerializeToString()
        print(example_str)
        str_len = len(example_str)

        writer.write(struct.pack('H', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))


if __name__ == '__main__':
    text_to_binary('data.txt', 'data.bin')

