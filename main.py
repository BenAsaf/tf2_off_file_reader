import tensorflow as tf
import trimesh
import os


MODELNET40_DOWNLOAD_URL = "http://modelnet.cs.princeton.edu/ModelNet40.zip"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")


def download_maybe(output_dir: str):
    data_dir = os.path.join(output_dir, "ModelNet40")
    if not os.path.exists(os.path.join(output_dir, "ModelNet40")):
        tf.keras.utils.get_file(
            "modelnet40.zip",
            MODELNET40_DOWNLOAD_URL,
            extract=True,
            cache_subdir=output_dir
        )
    return data_dir


def read_OFF_file(path):
    raw = tf.io.read_file(path)  # Read the data.

    raw = tf.strings.substr(raw, 3, tf.strings.length(raw))  # Substring and remove the "OFF"
    raw = tf.strings.strip(raw)  # Strip the extra spaces

    raw = tf.strings.regex_replace(raw, r"#.*\n$", "\n")  # Remove comments
    raw = tf.strings.split(raw, '\n')  # Split by lines.

    meta_data = tf.strings.to_number(input=tf.strings.split(raw[0], " "), out_type=tf.int32)
    num_verts, num_faces, num_edges = tf.split(meta_data, 3)
    num_verts = tf.reshape(num_verts, ())
    num_faces = tf.reshape(num_faces, ())
    # num_edges = tf.reshape(num_edges, ())  # Irrelevant in 'OFF' format

    start_idx_of_verts = 1  # First line is "OFF" (we removed it), second line is: "num_verts num_faces num_edges"
    end_idx_of_verts = start_idx_of_verts + num_verts
    start_idx_of_faces = end_idx_of_verts
    end_idx_of_faces = start_idx_of_faces + num_faces
    # start_idx_of_edges = end_idx_of_faces  # Irrelevant in 'OFF' format
    # end_idx_of_edges = start_idx_of_edges + num_edges  # Irrelevant in 'OFF' format

    vertices_raw = tf.strings.strip(raw[start_idx_of_verts:end_idx_of_verts])  # Remove extra spaces ' '
    faces_raw = tf.strings.strip(raw[start_idx_of_faces:end_idx_of_faces])
    # edges_raw = tf.strings.strip(raw[start_idx_of_edges:end_idx_of_edges])  # Irrelevant in 'OFF' format

    points = tf.strings.to_number(tf.strings.split(vertices_raw, " "), out_type=tf.float32).to_tensor()
    points = points - tf.reduce_mean(points, axis=0, keepdims=True)

    faces = tf.strings.to_number(tf.strings.split(faces_raw, " "), out_type=tf.int32).to_tensor()
    _, faces = tf.split(faces, axis=1, num_or_size_splits=(1, 3))  # Discard the first column which describes how many points.

    # edges = tf.strings.to_number(tf.strings.split(edges_raw, " "), out_type=tf.int32).to_tensor()  # Irrelevant in 'OFF' format

    return (points, faces), path


def main():
    global OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_dir = download_maybe(output_dir=OUTPUT_DIR)

    dataset = tf.data.Dataset.list_files(os.path.join(data_dir, "**", "**", "**.off"), shuffle=True)
    dataset = dataset.map(read_OFF_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.batch(1)
    dataset = dataset.prefetch(1)
    for x in dataset:
        (points, faces), path = x
        path = path.numpy().decode()
        tm = trimesh.Trimesh(vertices=points, faces=faces)
        tm.show(caption=path)




if __name__ == '__main__':
    main()