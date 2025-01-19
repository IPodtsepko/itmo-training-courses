drop table if exists Marks;
drop table if exists Courses;
drop table if exists Students;
drop table if exists Groups;
drop table if exists Subjects;
drop table if exists Teachers;

create table Groups
(
    group_id int not null,
    name char(6) not null,
    primary key (group_id)
);

create table Students
(
    student_id int not null,
    first_name varchar(30) not null,
    last_name varchar(30) not null,
    birthday date not null,
    phone char(11) not null,
    group_id int not null,
    primary key (student_id),
    constraint fk_group
        foreign key (group_id)
        references Groups (group_id)
);

create table Subjects
(
    subject_id int not null,
    name varchar(256) not null,
    primary key (subject_id)
);

create table Teachers
(
    teacher_id int not null,
    first_name varchar(30) not null,
    last_name varchar(30) not null,
    birthday date not null,
    phone char(11) not null,
    primary key (teacher_id)
);

create table Courses
(
    group_id int not null,
    subject_id int not null,
    teacher_id int not null,
    year int not null,
    term int not null,
    primary key
    (
        subject_id,
        group_id,
        teacher_id
    ),
    constraint fk_group
        foreign key (group_id)
        references Groups (group_id),
    constraint fk_subject
        foreign key (subject_id)
        references Subjects (subject_id),
    constraint fk_teacher
        foreign key (teacher_id)
        references Teachers (teacher_id)
);

create table Marks
(
    student_id int not null,
    subject_id int not null,
    group_id int not null,
    teacher_id int not null,
    mark int not null,
    primary key
    (
        student_id,
        subject_id,
        group_id,
        teacher_id
    ),
    constraint fk_student
        foreign key (student_id)
        references Students (student_id),
    constraint fk_course
        foreign key
        (
            subject_id,
            group_id,
            teacher_id
        )
        references Courses
        (
            subject_id,
            group_id,
            teacher_id
        )
);