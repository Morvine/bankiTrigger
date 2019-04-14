import tensorflow as tf

def model_custom(features, labels, mode, params):

    inputs = tf.layers.dense(inputs=features['x'], units=150, activation=tf.nn.relu)

    
    embeddings = tf.layers.dense(inputs=inputs, units=50)
    
    embedding_mean_norm = tf.reduce_mean(tf.norm(embeddings, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'embeddings': embeddings}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)

    loss = batch_hard_triplet_loss(labels, embeddings, margin=0.5, squared=False)


 
    with tf.variable_scope("metrics"):
        eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}



    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


    tf.summary.scalar('loss', loss)



    optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)
    global_step = tf.train.get_global_step()

    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
